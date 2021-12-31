#[cfg(feature = "blosc")]
use hdf5::filters::blosc_set_nthreads;
use std::env;
use hdf5::{File, H5Type, Result, Extent};
use ndarray::{arr1, s};
use project_a::utils::hdf5::*;

fn main() -> Result<()> {
    let vec = vec![10; 20];
    let args: Vec<String> = env::args().collect();
    let file = File::create(&args[1])?;
    let group = file.create_group("X")?;
    let data = ResizableVectorData::new(&group, "vec", 10000)?;
    data.extend(vec.into_iter())?;
    Ok(())
}