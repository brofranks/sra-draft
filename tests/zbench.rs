use std::ops::ControlFlow;
use std::time::Instant;
use sra::context::Context;
use sra::query::IntoQuery;
use sra::rel::Rel;
use sra::row::Row;

fn rel(cx: &Context, row: &Row, n: u64, k: u64, mul: u64) -> Rel {
    let mut r = Rel::new(cx, row.clone(), row.clone());
    r.insert_rows((0..n).map(|i| (i as u16, ((i * mul + 7) % k) as u16)));
    r
}

#[test]
fn zbench() {
    let only = std::env::var("ZB").ok();
    for backend in ["sparse", "bdd"] {
        if only.as_deref().is_none_or(|b| b == backend) {
            bench(backend);
        }
    }
}

fn bench(backend: &str) {
    let mut cx = Context::default();
    let sort = cx.sorts().bv(16);
    let column = cx.bind(backend, sort.scalar());
    let row = Row::new(&cx, sort, [column].into(), 0);
    let a = rel(&cx, &row, 100_000, 20_000, 7);
    let b = rel(&cx, &row, 100_000, 20_000, 13);
    let t = Instant::now();
    let c = (&a).select(&b).collect();
    let collect = t.elapsed();
    let t = Instant::now();
    let n = if std::env::var("ZB").is_ok() { None } else { Some((&a).select(&b).count()) };
    let count = t.elapsed();
    let t = Instant::now();
    let mut seen = 0u64;
    let _ = (&a).select(&b).drive(&mut |_, _| { seen += 1; ControlFlow::<()>::Continue(()) });
    let drive = t.elapsed();
    let t = Instant::now();
    for _ in 0..2_000 {
        assert!(!(&a).select(&b).is_empty());
    }
    let empty = t.elapsed();
    let t = Instant::now();
    for _ in 0..2_000 {
        assert!(!(&a).is_empty());
    }
    let src_empty = t.elapsed();
    println!("BENCH {backend} collect={collect:?} count={count:?} ({n:?}, rows={}) drive={drive:?} ({seen}) is_empty(join)x2000={empty:?} is_empty(source)x2000={src_empty:?}", c.count());
}
