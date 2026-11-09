use std::rc::Rc;

use argh::FromArgs;
use mpi::{
    self,
    traits::{Communicator, CommunicatorCollectives, Root},
};
use rayon::{
    ThreadPoolBuilder,
    iter::{
        IndexedParallelIterator, IntoParallelRefIterator, IntoParallelRefMutIterator,
        ParallelIterator,
    },
};

/// Gas particles under heavy piston simulator
#[derive(FromArgs)]
struct Args {
    /// free fall acceleration
    #[argh(option, default = "9.8", short = 'g')]
    freefall: f64,

    /// mass of the piston
    #[argh(option, default = "1.0", short = 'M')]
    piston_mass: f64,

    /// mass of a particle
    #[argh(option, default = "1e-3", short = 'm')]
    particle_mass: f64,

    /// duration of a simulation step
    #[argh(option, default = "1e-3", short = 't')]
    time_step: f64,

    /// duration of an observation step
    #[argh(option, default = "0.1", short = 'o')]
    observation_step: f64,

    /// total simulation time
    #[argh(option, default = "10.0", short = 'T')]
    total_simulation_time: f64,

    /// starting height of the piston
    #[argh(option, default = "f64::NAN", short = 'h')]
    height: f64,

    /// amount of particles per rank
    #[argh(option, default = "1000", short = 'n')]
    amount: usize,

    /// initial speed of particles
    #[argh(option, default = "10.0", short = 'v')]
    speed: f64,

    /// hardness of the material
    #[argh(option, default = "10.0", short = 'k')]
    hardness: f64,

    /// parallel threads
    #[argh(option, default = "1", short = 'p')]
    threads: usize,
}

fn main() {
    let (universe, support) = mpi::initialize_with_threading(mpi::Threading::Funneled).unwrap();
    let world = Rc::new(universe.world());
    let size = world.size() as f64;
    let rank = world.rank();
    if support < mpi::Threading::Funneled {
        return;
    }

    let mut args: Args = argh::from_env();
    if args.height.is_nan() {
        args.height = args.particle_mass * args.amount as f64 * args.speed * args.speed * size
            / (args.piston_mass * args.freefall)
    }

    let [steps, inner_steps] = {
        let mut steps_buffer = [
            (args.total_simulation_time / args.observation_step).round() as u64,
            (args.observation_step / args.time_step).round() as u64,
        ];
        world.process_at_rank(0).broadcast_into(&mut steps_buffer);
        steps_buffer
    };

    ThreadPoolBuilder::new()
        .num_threads(args.threads)
        .build_global()
        .unwrap();

    let mut simulation = Simulation::new(args, inner_steps, world.clone());

    let report = simulation.report();
    if rank == 0 {
        report.print_csv();
    }
    for _ in 0..steps {
        simulation.observation_step();
        let report = simulation.report();
        if rank == 0 {
            report.print_csv();
        }
    }
}

struct Simulation {
    inner_steps: u64,
    current_step: u64,
    args: Args,
    accelerations: Vec<f64>,
    new_accelerations: Vec<f64>,
    positions: Vec<f64>,
    speeds: Vec<f64>,
    piston_positions: [f64; 3],
    communicator: Rc<dyn Communicator>,
}

impl Simulation {
    fn new(args: Args, inner_steps: u64, communicator: Rc<dyn Communicator>) -> Self {
        let rank = communicator.rank() as f64;
        let size = communicator.size() as f64;
        let amount_with_piston = args.amount + 1;
        let accelerations = vec![0.0; amount_with_piston];
        let new_accelerations = vec![0.0; amount_with_piston];
        let mut speeds = vec![0.0];
        speeds.extend((0..args.amount).map(|i| {
            let sign = if i & 1 == 0 { 1.0 } else { -1.0 };
            sign * args.speed
        }));

        let mut positions = vec![args.height];
        positions.extend((0..args.amount).map(|i| {
            let ratio = (i as f64) / (args.amount as f64 * size + 1.0) + rank / size;
            args.height * ratio
        }));
        let piston_positions = [args.height; 3];
        let mut res = Self {
            args,
            communicator,
            inner_steps,
            current_step: 0,
            accelerations,
            new_accelerations,
            positions,
            speeds,
            piston_positions,
        };
        res.calculate_new_accelerations();
        res
    }
    fn observation_step(&mut self) {
        for _ in 0..self.inner_steps {
            self.step();
        }
        self.current_step += self.inner_steps;
        let [_, y, z] = self.piston_positions;
        self.piston_positions = [y, z, self.positions[0]];
    }
    fn step(&mut self) {
        self.swap_accelerations();
        self.calculate_new_positions();
        self.calculate_new_accelerations();
        self.calculate_new_speeds();
    }
    fn swap_accelerations(&mut self) {
        std::mem::swap(&mut self.accelerations, &mut self.new_accelerations);
    }
    fn calculate_new_positions(&mut self) {
        let time_step = self.args.time_step;
        let time_step_squared = time_step * time_step;
        self.positions
            .par_iter_mut()
            .zip(self.accelerations.par_iter().copied())
            .zip(self.speeds.par_iter().copied())
            .for_each(|((p, a), v)| *p += time_step * v + 0.5 * a * time_step_squared);
    }
    fn calculate_new_accelerations(&mut self) {
        let piston_position = self.positions[0];
        self.new_accelerations[1..]
            .par_iter_mut()
            .zip(self.positions[1..].par_iter().copied())
            .for_each(|(a, p)| {
                *a = if p < 0.0 {
                    -p * self.args.hardness
                } else if p > piston_position {
                    (piston_position - p) * self.args.hardness
                } else {
                    0.0
                }
            });

        let local_force_on_piston = -self.new_accelerations[1..]
            .par_iter()
            .copied()
            .filter(|f| *f < 0.0)
            .sum::<f64>();
        let mut force_on_piston = 0.0;
        self.communicator.all_reduce_into(
            &local_force_on_piston,
            &mut force_on_piston,
            mpi::collective::SystemOperation::sum(),
        );

        self.new_accelerations[1..]
            .par_iter_mut()
            .for_each(|f| *f /= self.args.particle_mass);

        let piston_acceleration = force_on_piston / self.args.piston_mass - self.args.freefall;
        self.new_accelerations[0] = piston_acceleration;
    }
    fn calculate_new_speeds(&mut self) {
        self.speeds
            .par_iter_mut()
            .zip(self.accelerations.par_iter().copied())
            .zip(self.new_accelerations.par_iter().copied())
            .for_each(|((v, a), n)| *v += 0.5 * (a + n) * self.args.time_step);
    }
    fn report(&self) -> Report {
        let local_particles_speed_squared: f64 =
            self.speeds[1..].par_iter().copied().map(|v| v * v).sum();
        let mut particles_speed_squared = 0.0;
        self.communicator.all_reduce_into(
            &local_particles_speed_squared,
            &mut particles_speed_squared,
            mpi::collective::SystemOperation::sum(),
        );
        let particles_energy = 0.5 * self.args.particle_mass * particles_speed_squared;
        let piston_speed = self.speeds[0];
        let piston_position = self.positions[0];
        let piston_energy = self.args.piston_mass * self.args.freefall * piston_position
            + 0.5 * self.args.piston_mass * piston_speed * piston_speed;
        let piston_force = {
            let [x, y, z] = self.piston_positions;
            self.args.piston_mass
                * ((-x + y + y - z) / self.args.observation_step.powi(2) + self.args.freefall)
        };
        let projected_force = self.args.particle_mass * particles_speed_squared / piston_position;
        let force_deviation = (piston_force - projected_force) / (projected_force.abs() + 1e-8);
        let time = self.current_step as f64 * self.args.time_step;
        Report {
            time,
            piston_position,
            piston_energy,
            particles_energy,
            force_deviation,
        }
    }
}

struct Report {
    time: f64,
    piston_position: f64,
    piston_energy: f64,
    particles_energy: f64,
    force_deviation: f64,
}

impl Report {
    fn print_csv(&self) {
        println!(
            "{:.5},{:.5},{:.5},{:.5},{:.5},{:.5}",
            self.time,
            self.piston_position,
            self.piston_energy + self.particles_energy,
            self.piston_energy,
            self.particles_energy,
            self.force_deviation,
        )
    }
}
