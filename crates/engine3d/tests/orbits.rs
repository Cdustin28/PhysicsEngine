//! N-body orbits checked against Kepler and the conservation laws.

use engine3d::orbits::{
    AU, DAY, G, Kind, M_EARTH, M_SUN, OrbitSystem, R_EARTH, YEAR, solar_system, sun_only,
};

/// Run until `seconds` have passed, as the editor does (≤ 1 day per call).
fn run(s: &mut OrbitSystem, seconds: f64) {
    let end = s.time + seconds;
    while s.time < end - 1e-6 {
        let chunk = (end - s.time).min(DAY);
        s.advance(chunk, 100_000);
    }
}

/// Average period over `orbits` orbits, from the angle body `i` sweeps
/// around body `around` (unwrapped, interpolated within the last step).
fn measured_period(s: &mut OrbitSystem, i: usize, around: usize, orbits: f64) -> f64 {
    let angle = |s: &OrbitSystem| {
        let d = s.bodies[i].position - s.bodies[around].position;
        d.z.atan2(d.x)
    };
    let (mut prev, mut swept, start) = (angle(s), 0.0_f64, s.time);
    let target = orbits * std::f64::consts::TAU;
    loop {
        let dt = s.stable_dt();
        s.step(dt);
        let a = angle(s);
        let mut d = a - prev;
        d -= std::f64::consts::TAU * (d / std::f64::consts::TAU).round();
        if swept.abs() + d.abs() >= target {
            // Interpolate the final partial step.
            let frac = (target - swept.abs()) / d.abs();
            return (s.time - dt + frac * dt - start) / orbits;
        }
        swept += d;
        prev = a;
    }
}

#[test]
fn keplers_third_law() {
    // Light test planets (no mutual pull worth measuring) from 0.4 to 30 AU:
    // T = 2π √(a³ / G(M + m)).
    for a_au in [0.4, 1.0, 5.2, 30.0] {
        let mut s = sun_only();
        let p = s.add_planet("Test", 1e20, 1e6, [255, 255, 255], a_au * AU, 0.0);
        let mu = G * (M_SUN + 1e20);
        let expected = std::f64::consts::TAU * ((a_au * AU).powi(3) / mu).sqrt();
        let t = measured_period(&mut s, p, 0, 2.0);
        // Leapfrog at 256 steps per orbit runs long by (2π/256)²/3 = 2.0e-4.
        assert!(
            (t / expected - 1.0).abs() < 3e-4,
            "a = {a_au} AU: T = {t}, expected {expected}"
        );
    }
}

#[test]
fn earths_year_and_the_moons_month() {
    let mut s = solar_system();
    let earth = s.bodies.iter().position(|b| b.name == "Earth").unwrap();
    let moon = s.bodies.iter().position(|b| b.name == "Moon").unwrap();
    let el = s.elements(earth).unwrap();
    // Sidereal year 365.256 d; the two-body period of the Earth–Moon
    // barycenter from its current state.
    let year = el.period.unwrap() / DAY;
    assert!((year - 365.256).abs() < 0.5, "Earth's year {year} d");
    assert!(
        (el.eccentricity - 0.0167).abs() < 0.002,
        "e = {}",
        el.eccentricity
    );
    // Sidereal month 27.32 d.
    let month = s.elements(moon).unwrap().period.unwrap() / DAY;
    assert!((month - 27.32).abs() < 0.3, "Moon's month {month} d");
    // Still bound to Earth after a year of everything pulling on it.
    run(&mut s, YEAR);
    let el = s.elements(moon).unwrap();
    assert!(el.period.is_some() && el.eccentricity < 0.1, "{el:?}");
    let d = (s.bodies[moon].position - s.bodies[earth].position).length();
    assert!((3.5e8..4.1e8).contains(&d), "Moon {d} m from Earth");
}

#[test]
fn solar_system_conserves_energy_and_angular_momentum() {
    let mut s = solar_system();
    let (e0, l0) = (s.energy(), s.angular_momentum());
    run(&mut s, 20.0 * YEAR);
    let (e1, l1) = (s.energy(), s.angular_momentum());
    let de = ((e1 - e0) / e0).abs();
    let dl = (l1 - l0).length() / l0.length();
    assert!(de < 1e-6, "energy drift {de}");
    assert!(dl < 1e-9, "angular momentum drift {dl}");
}

#[test]
fn moving_a_planet_keeps_its_moons() {
    let mut s = solar_system();
    let jupiter = s.bodies.iter().position(|b| b.name == "Jupiter").unwrap();
    let io = s.bodies.iter().position(|b| b.name == "Io").unwrap();
    let before = s.elements(io).unwrap();
    s.set_distance(jupiter, 2.0 * AU);
    let el = s.elements(jupiter).unwrap();
    assert!((el.distance / (2.0 * AU) - 1.0).abs() < 1e-9);
    assert!(el.eccentricity < 1e-6, "circular: e = {}", el.eccentricity);
    let after = s.elements(io).unwrap();
    assert!((after.semi_major_axis / before.semi_major_axis - 1.0).abs() < 1e-9);
    // And it stays on its new orbit for a while.
    run(&mut s, YEAR);
    let el = s.elements(jupiter).unwrap();
    assert!((el.distance / (2.0 * AU) - 1.0).abs() < 0.01, "{el:?}");
}

#[test]
fn moving_a_moon_leaves_its_planet_alone() {
    let mut s = solar_system();
    let earth = s.bodies.iter().position(|b| b.name == "Earth").unwrap();
    let moon = s.bodies.iter().position(|b| b.name == "Moon").unwrap();
    let before = s.elements(earth).unwrap();
    s.set_distance(moon, 2e8);
    let el = s.elements(moon).unwrap();
    assert!((el.distance / 2e8 - 1.0).abs() < 1e-9 && el.eccentricity < 1e-6);
    let after = s.elements(earth).unwrap();
    assert!((after.semi_major_axis / before.semi_major_axis - 1.0).abs() < 1e-9);
}

#[test]
fn adding_a_moon_gives_a_circular_orbit_at_the_asked_distance() {
    let mut s = sun_only();
    let p = s.add_planet("World", M_EARTH, R_EARTH, [0, 0, 255], AU, 0.0);
    let m = s.add_moon(
        p,
        "Moonlet",
        1e21,
        5e5,
        [200, 200, 200],
        engine3d::orbits::OrbitSpec::circular(1e8, 0.0),
    );
    assert_eq!(s.bodies[m].kind, Kind::Moon);
    let el = s.elements(m).unwrap();
    assert!((el.distance / 1e8 - 1.0).abs() < 1e-12 && el.eccentricity < 1e-9);
    // The planet's (group's) orbit around the Sun didn't change.
    let el = s.elements(p).unwrap();
    assert!(
        (el.distance / AU - 1.0).abs() < 1e-9 && el.eccentricity < 1e-6,
        "{el:?}"
    );
}

#[test]
fn collisions_merge_and_conserve_mass_and_momentum() {
    let mut s = sun_only();
    let a = s.add_planet("A", M_EARTH, R_EARTH, [0, 0, 255], AU, 0.0);
    // Put B just outside A, falling straight onto it.
    let b = s.add_moon(
        a,
        "B",
        0.1 * M_EARTH,
        0.5 * R_EARTH,
        [255, 0, 0],
        engine3d::orbits::OrbitSpec::circular(3.0 * R_EARTH, 0.0),
    );
    s.bodies[b].velocity = s.bodies[a].velocity;
    let mass: f64 = s.bodies.iter().map(|x| x.mass).sum();
    let momentum: glam::DVec3 = s.bodies.iter().map(|x| x.mass * x.velocity).sum();
    let mut merged = Vec::new();
    for _ in 0..2000 {
        merged.extend(s.advance(60.0, 1000).mergers);
        if !merged.is_empty() {
            break;
        }
    }
    assert_eq!(merged, vec![("B".to_string(), "A".to_string())]);
    assert_eq!(s.bodies.len(), 2);
    let mass2: f64 = s.bodies.iter().map(|x| x.mass).sum();
    let momentum2: glam::DVec3 = s.bodies.iter().map(|x| x.mass * x.velocity).sum();
    assert!((mass2 / mass - 1.0).abs() < 1e-12);
    assert!((momentum2 - momentum).length() < 1e-6 * M_EARTH * 3e4);
}

#[test]
fn removing_a_planet_removes_its_moons() {
    let mut s = solar_system();
    let n = s.bodies.len();
    let jupiter = s.bodies.iter().position(|b| b.name == "Jupiter").unwrap();
    let gone = s.remove(jupiter);
    assert_eq!(gone.len(), 5);
    assert_eq!(s.bodies.len(), n - 5);
    // Parent links still point at the right bodies.
    for b in &s.bodies {
        if let Some(p) = b.parent {
            let host = &s.bodies[p];
            match b.kind {
                Kind::Moon => assert_eq!(host.kind, Kind::Planet),
                Kind::Planet => assert_eq!(host.kind, Kind::Star),
                Kind::Star => unreachable!(),
            }
        }
    }
    let triton = s.bodies.iter().find(|b| b.name == "Triton").unwrap();
    assert_eq!(s.bodies[triton.parent.unwrap()].name, "Neptune");
}

#[test]
fn round_trips_through_json() {
    let s = solar_system();
    let json = serde_json::to_string(&s).unwrap();
    let back: OrbitSystem = serde_json::from_str(&json).unwrap();
    assert_eq!(back.bodies.len(), s.bodies.len());
    assert_eq!(back.bodies[3].position, s.bodies[3].position);
}
