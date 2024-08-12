# **Spatial Block**

The Spatial Block focuses on capturing spatial relationships and interactions between nodes in a graph. The following neural networks are effective for this block:

- **Graph Convolutional Networks (GCNs)**: GCNs are designed to aggregate information from neighboring nodes and can effectively capture local graph structures.

    - **Standard GCN**: Aggregates information from neighbors using a fixed convolutional kernel.

- **Graph Attention Networks (GATs)**: GATs use attention mechanisms to weigh the importance of neighboring nodes, allowing for more flexible and adaptive aggregation.

    - **Self-Attention Mechanism**: Weighs the influence of each neighbor dynamically based on their relevance.

- **Graph Isomorphism Networks (GINs)**: GINs are designed to capture more expressive node representations by using more powerful aggregation functions compared to traditional GCNs.

    - **Sum Aggregation**: Utilizes a sum-based aggregation approach to improve expressiveness.

- **Spatial Graph Convolutional Networks (SGCNs)**: SGCNs focus on spatial relationships in graph structures, often incorporating techniques to capture spatial features more effectively.
