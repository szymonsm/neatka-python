from __future__ import annotations
from typing import TYPE_CHECKING
from node import Node

if TYPE_CHECKING:  # pragma: no cover
    from genome import Genome

class ConnectionHistory():
    """
    Connection History

    Represents a connection history entry in the NEAT algorithm, tracking the innovation numbers
    and nodes involved in a connection.

    Attributes:
    - from_node (Node): The source node of the connection.
    - to_node (Node): The target node of the connection.
    - innovation_nb (int): The innovation number uniquely identifying this connection.

    Methods:
    - matches(from_node: Node, to_node: Node) -> bool: Returns whether the genome matches the original genome and the connection is between the same nodes

    """

    def __init__(self, from_node: Node, to_node: Node, innovation_nb: int) -> None:
        """
        Initialize a ConnectionHistory instance.

        Args:
        - from_node (Node): The source node of the connection.
        - to_node (Node): The target node of the connection.
        - innovation_nb (int): The innovation number uniquely identifying this connection.

        """
        self.from_node = from_node
        self.to_node = to_node
        self.innovation_nb = innovation_nb

    def matches(self, from_node: Node, to_node: Node):
        """
        Returns whether the genome matches the original genome and the connection is between the same nodes

        Args:
        - from_node (Node): The source node of the connection to check.
        - to_node (Node): The target node of the connection to check.

        Returns:
        - bool: True if the genome and connection nodes match the history entry, False otherwise.

        """
        return from_node.id == self.from_node.id and to_node.id == self.to_node.id
