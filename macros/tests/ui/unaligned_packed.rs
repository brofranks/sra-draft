#[derive(sra::Schema)]
#[repr(C, packed)]
struct Packet {
    tag: u8,
    value: u32,
}

fn main() {}
