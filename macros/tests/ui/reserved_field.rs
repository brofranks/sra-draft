#[derive(sra::Schema)]
struct Reserved {
    sort: u8,
}

#[derive(sra::Schema)]
struct AlsoReserved {
    r#constructor: u8,
}

fn main() {}
