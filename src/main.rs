mod format;
mod graph;

use crate::format::{Model, Src};
use anyhow::Result;
use clap::Parser;
use format::LayerRef;
use graph::{Graph, MarkedLayer};
use itertools::{chain, izip};
use std::{
    fs::{self, File},
    io::{BufWriter, Write},
    path::PathBuf,
};

#[derive(Parser)]
struct Opts {
    pub input_file: PathBuf,
    pub output_file: PathBuf,
}

fn main() -> Result<()> {
    let opts = Opts::parse();

    let text = fs::read_to_string(&opts.input_file)?;
    let model: Model = serde_yaml::from_str(&text)?;

    let num_backbone_layers = model.backbone.len();

    let mut iter = model.backbone.into_iter();
    let first_layer = {
        let layer = iter.next().unwrap();
        assert!(matches!(layer.from, Src::Single(LayerRef::Relative(1))));
        MarkedLayer {
            from: vec![],
            layer,
        }
    };
    let backbone_layers = izip!(1.., iter).map(|(layer_id, layer)| {
        let from: Vec<_> = match &layer.from {
            Src::Single(LayerRef::Relative(offset)) => vec![layer_id - offset],
            Src::Single(LayerRef::Absolute(pos)) => vec![*pos],
            Src::Multi(indices) => indices
                .iter()
                .map(|&id| match id {
                    LayerRef::Relative(offset) => layer_id - offset,
                    LayerRef::Absolute(pos) => pos,
                })
                .collect(),
        };

        MarkedLayer { from, layer }
    });

    let head_layers = izip!(num_backbone_layers.., model.head).map(|(layer_id, layer)| {
        let from: Vec<_> = match &layer.from {
            Src::Single(LayerRef::Relative(offset)) => vec![layer_id - offset],
            Src::Single(LayerRef::Absolute(pos)) => vec![*pos],
            Src::Multi(indices) => indices
                .iter()
                .map(|&id| match id {
                    LayerRef::Relative(offset) => layer_id - offset,
                    LayerRef::Absolute(pos) => pos,
                })
                .collect(),
        };

        MarkedLayer { from, layer }
    });

    let layers: Vec<_> = chain!([first_layer], backbone_layers, head_layers).collect();
    let graph = Graph::new(layers);

    let mut writer = BufWriter::new(File::create(&opts.output_file)?);
    dot::render(&graph, &mut writer)?;
    writer.flush()?;

    Ok(())
}
