use std::borrow::Cow;

use dot::{Edges, GraphWalk, Id, Labeller, Nodes, LabelText};

use crate::format::Layer;

#[derive(Debug, Clone)]
pub struct Graph {
    nds: Vec<Nd>,
    eds: Vec<Ed>,
    layers: Vec<MarkedLayer>,
}

impl Graph {
    pub fn new(layers: Vec<MarkedLayer>) -> Self {
        let nds: Vec<_> = (0..layers.len()).collect();
        let eds: Vec<_> = layers
            .iter()
            .enumerate()
            .flat_map(|(dst, layer)| layer.from.iter().map(move |&src| (src, dst)))
            .collect();

        Self {
            layers,
            nds: nds.into(),
            eds: eds.into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MarkedLayer {
    pub from: Vec<usize>,
    pub layer: Layer,
}

pub type Nd = usize;
pub type Ed = (usize, usize);

impl<'a> Labeller<'a, Nd, Ed> for Graph {
    fn graph_id(&'a self) -> Id<'a> {
        Id::new("yolov7").unwrap()
    }

    fn node_id(&'a self, nid: &Nd) -> Id<'a> {
        Id::new(format!("N{nid}")).unwrap()
    }

    fn node_label(&'a self, &nid: &Nd) -> LabelText<'a> {
        let layer = &self.layers[nid];
        LabelText::label(format!("({}) {}", nid, layer.layer.kind))
    }
}

impl<'a> GraphWalk<'a, Nd, Ed> for Graph {
    fn nodes(&'a self) -> Nodes<'a, Nd> {
        Cow::from(&self.nds)
    }

    fn edges(&'a self) -> Edges<'a, Ed> {
        Cow::from(&self.eds)
    }

    fn source(&'a self, &(src, _): &Ed) -> Nd {
        src
    }

    fn target(&'a self, &(_, dst): &Ed) -> Nd {
        dst
    }
}
