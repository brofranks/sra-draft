#[derive(sra::Schema)]
struct Bounded
where
    u8: Copy,
{
    a: u8,
}

fn main() {}
