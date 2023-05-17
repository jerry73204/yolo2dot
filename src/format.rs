use itertools::Itertools;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SerializedModel", into = "SerializedModel")]
pub struct Model {
    pub nc: usize,
    pub depth_multiple: f64,
    pub width_multiple: f64,
    pub anchors: Vec<AnchorList>,
    pub backbone: Vec<Layer>,
    pub head: Vec<Layer>,
}

impl From<Model> for SerializedModel {
    fn from(from: Model) -> Self {
        let Model {
            nc,
            depth_multiple,
            width_multiple,
            anchors,
            backbone,
            head,
        } = from;

        let anchors: Vec<_> = anchors
            .into_iter()
            .map(SerializedAnchorList::from)
            .collect();
        let backbone: Vec<_> = backbone.into_iter().map(SerializedLayer::from).collect();
        let head: Vec<_> = head.into_iter().map(SerializedLayer::from).collect();

        Self {
            nc,
            depth_multiple,
            width_multiple,
            anchors,
            backbone,
            head,
        }
    }
}

impl From<SerializedModel> for Model {
    fn from(from: SerializedModel) -> Self {
        let SerializedModel {
            nc,
            depth_multiple,
            width_multiple,
            anchors,
            backbone,
            head,
        } = from;

        let anchors: Vec<_> = anchors.into_iter().map(AnchorList::from).collect();
        let backbone: Vec<_> = backbone.into_iter().map(Layer::from).collect();
        let head: Vec<_> = head.into_iter().map(Layer::from).collect();

        Self {
            nc,
            depth_multiple,
            width_multiple,
            anchors,
            backbone,
            head,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedModel {
    pub nc: usize,
    pub depth_multiple: f64,
    pub width_multiple: f64,
    pub anchors: Vec<SerializedAnchorList>,
    pub backbone: Vec<SerializedLayer>,
    pub head: Vec<SerializedLayer>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedAnchorList(Vec<usize>);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SerializedAnchorList", into = "SerializedAnchorList")]
pub struct AnchorList(Vec<WH>);

impl From<AnchorList> for SerializedAnchorList {
    fn from(from: AnchorList) -> Self {
        let AnchorList(anchors) = from;
        let values: Vec<_> = anchors.into_iter().flat_map(|WH { w, h }| [w, h]).collect();
        Self(values)
    }
}

impl From<SerializedAnchorList> for AnchorList {
    fn from(from: SerializedAnchorList) -> Self {
        let SerializedAnchorList(values) = from;
        assert!(values.len() % 2 == 0);
        let sizes: Vec<_> = values
            .into_iter()
            .tuple_windows()
            .map(|(w, h)| WH { w, h })
            .collect();
        Self(sizes)
    }
}

#[derive(Debug, Clone, Copy)]
pub struct WH {
    pub w: usize,
    pub h: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SerializedLayer(pub SerializedSrc, pub usize, pub LayerKind, pub Vec<Param>);

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SerializedLayer", into = "SerializedLayer")]
pub struct Layer {
    pub from: Src,
    pub multiple: usize,
    pub kind: LayerKind,
    pub params: Vec<Param>,
}

impl From<SerializedLayer> for Layer {
    fn from(from: SerializedLayer) -> Self {
        let SerializedLayer(from, multiple, kind, params) = from;
        Self {
            from: from.into(),
            multiple,
            kind,
            params,
        }
    }
}

impl From<Layer> for SerializedLayer {
    fn from(from: Layer) -> Self {
        let Layer {
            from,
            multiple: to,
            kind,
            params,
        } = from;
        Self(from.into(), to, kind, params)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum SerializedSrc {
    Single(isize),
    Multi(Vec<isize>),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(from = "SerializedSrc", into = "SerializedSrc")]
pub enum Src {
    Single(LayerRef),
    Multi(Vec<LayerRef>),
}

impl From<SerializedSrc> for Src {
    fn from(from: SerializedSrc) -> Self {
        match from {
            SerializedSrc::Single(index) => Self::Single(LayerRef::from_index(index)),
            SerializedSrc::Multi(indices) => {
                Self::Multi(indices.into_iter().map(LayerRef::from_index).collect())
            }
        }
    }
}

impl From<Src> for SerializedSrc {
    fn from(from: Src) -> Self {
        match from {
            Src::Single(index) => Self::Single(index.to_index()),
            Src::Multi(indices) => Self::Multi(indices.iter().map(LayerRef::to_index).collect()),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LayerRef {
    Relative(usize),
    Absolute(usize),
}

impl LayerRef {
    pub fn to_index(&self) -> isize {
        match *self {
            LayerRef::Relative(offset) => -(offset as isize),
            LayerRef::Absolute(pos) => pos as isize,
        }
    }

    pub fn from_index(index: isize) -> Self {
        if index >= 0 {
            Self::Absolute(index as usize)
        } else {
            Self::Relative(-index as usize)
        }
    }
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, Serialize, Deserialize, strum::Display)]
pub enum LayerKind {
    autoShape,
    Bottleneck,
    BottleneckCSPA,
    BottleneckCSPB,
    BottleneckCSPC,
    Chuncat,
    Classify,
    Concat,
    Contract,
    Conv,
    ConvBN,
    Detections,
    DownC,
    Expand,
    Focus,
    Foldcut,
    Ghost,
    GhostConv,
    GhostCSPA,
    GhostCSPB,
    GhostCSPC,
    GhostSPPCSPC,
    GhostStem,
    ImplicitA,
    ImplicitM,
    Mlp,
    Mlp_v2,
    MP,
    NMS,
    OREPA_3x3_RepConv,
    ReOrg,
    RepBottleneck,
    RepBottleneckCSPA,
    RepBottleneckCSPB,
    RepBottleneckCSPC,
    RepConv,
    RepConv_OREPA,
    RepRes,
    RepResCSPA,
    RepResCSPB,
    RepResCSPC,
    RepResX,
    RepResXCSPA,
    RepResXCSPB,
    RepResXCSPC,
    Res,
    ResCSPA,
    ResCSPB,
    ResCSPC,
    ResX,
    ResXCSPA,
    ResXCSPB,
    ResXCSPC,
    RobustConv,
    RobustConv2,
    Shortcut,
    SP,
    SPP,
    SPPCSPC,
    SPPF,
    ST2CSPA,
    ST2CSPB,
    ST2CSPC,
    STCSPA,
    STCSPB,
    STCSPC,
    Stem,
    SwinTransformer2Block,
    SwinTransformerBlock,
    SwinTransformerLayer,
    SwinTransformerLayer_v2,
    TransformerBlock,
    TransformerLayer,
    WindowAttention,
    WindowAttention_v2,
    Detect,
    IAuxDetect,
    IBin,
    IDetect,
    IKeypoint,
    #[serde(rename = "nn.Upsample")]
    nnUpsample,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Param {
    None,
    Value(isize),
    Text(String),
}
