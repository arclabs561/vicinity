//! Tree-based ANN methods.

#[cfg(any(
    feature = "kdtree",
    feature = "balltree",
    feature = "rptree",
    feature = "kmeans_tree"
))]
pub(crate) mod persistence;

#[cfg(feature = "rptree")]
pub mod rp_forest; // Random Projection Forest (RP-tree family)

#[cfg(feature = "kdtree")]
pub mod kdtree;

#[cfg(feature = "balltree")]
pub mod balltree;

#[cfg(feature = "rptree")]
pub mod random_projection;

#[cfg(feature = "kmeans_tree")]
pub mod kmeans_tree;
