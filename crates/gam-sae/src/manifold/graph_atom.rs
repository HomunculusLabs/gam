/// Named-shape compression certified after the graph has been learned.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphCompressionKind {
    Circle,
    Interval,
    FiniteSet,
    /// A contractible bounded surface (`χ = 1`, orientable, `b₁ = 0`, `b₂ = 0`):
    /// the topological type of a sampled sheet or disk. This is what a swiss roll
    /// glues to — the fold unrolls into one flat chart with no handle and no
    /// closed 2-cycle (#2280 acceptance: "swiss roll → sheet").
    Disk,
    Cylinder,
    /// The non-orientable bounded surface with one boundary loop (`χ = 0`,
    /// non-orientable, `b₁ = 1`, `b₂ = 0`): a cylinder's orientation cocycle with
    /// a single sign reversal around the loop. Recognized by the orientability
    /// certificate — the "half-twist as a discrete sign" (#2280 acceptance:
    /// "Möbius band → holonomy sign detected").
    MobiusStrip,
    Torus,
    Sphere,
    ProjectivePlane,
    KleinBottle,
    Graph,
}
