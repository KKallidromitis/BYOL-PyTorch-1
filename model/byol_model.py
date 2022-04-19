#-*- coding:utf-8 -*-
# Nearest neighbor code from https://github.com/vturrisi/solo-learn/blob/main/solo/methods/nnclr.py
import torch
from .basic_modules import EncoderwithProjection, Predictor
from utils.distributed_utils import gather_from_all

class BYOLModel(torch.nn.Module):
    def __init__(self, config):
        super().__init__()

        # online network
        self.online_network = EncoderwithProjection(config)

        # target network
        self.target_network = EncoderwithProjection(config)

        # predictor
        self.predictor = Predictor(config)

        self._initializes_target_network()

        self.queue_size = config['model']['memory_size']
        print(f"Using Nearest Neighbors with Queue size {self.queue_size}")
        # setup queue (For Storing Random Targets)
        self.register_buffer('queue', torch.randn(self.queue_size, config['model']['projection']['output_dim']))
        # normalize the queue embeddings
        self.queue = torch.nn.functional.normalize(self.queue, dim=1)
        # setup the queue pointer
        self.register_buffer('queue_ptr', torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _initializes_target_network(self):
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False     # not update by gradient

    @torch.no_grad()
    def _update_target_network(self, mm):
        """Momentum update of target network"""
        for param_q, param_k in zip(self.online_network.parameters(), self.target_network.parameters()):
            param_k.data.mul_(mm).add_(1. - mm, param_q.data)

    @torch.no_grad()
    def dequeue_and_enqueue(self, z: torch.Tensor):
        """Adds new samples and removes old samples from the queue in a fifo manner.
        Args:
            z (torch.Tensor): batch of projected features.
        """

        z = gather_from_all(z)
        
        batch_size = z.shape[0]

        ptr = int(self.queue_ptr)  # type: ignore
        assert self.queue_size % batch_size == 0

        self.queue[ptr : ptr + batch_size, :] = z
        ptr = (ptr + batch_size) % self.queue_size

        self.queue_ptr[0] = ptr  # type: ignore

    @torch.no_grad()
    def find_nn(self, z: torch.Tensor):
        """Finds the nearest neighbor of a sample.
        Returns idx, nearest-neighbor's embedding
        Args:
            z (torch.Tensor): a batch of projected features.
        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                indices and projected features of the nearest neighbors.
        """

        idx = (z @ self.queue.T).max(dim=1)[1]
        nn = self.queue[idx]
        return idx, nn

    def forward(self, target_to_enqeue):
        self.dequeue_and_enqueue(target_to_enqeue)

    def forward(self, view1, view2, mm):
        batch_size = view1.shape[0]

        # online network forward
        q = self.predictor(self.online_network(torch.cat([view1, view2], dim=0)))

        # target network forward
        with torch.no_grad():
            self._update_target_network(mm)
            target_z = self.target_network(torch.cat([view1, view2], dim=0)).detach().clone()

        # return q, target_z

        # Nearest Neighbor
        # Get nearest neighbor (using target network embeddings as per "With a Little Help From Your Friends...")
        _, nn1_target = self.find_nn(target_z[:batch_size])
        _, nn2_target = self.find_nn(target_z[batch_size:])

        # Update queue
        self.dequeue_and_enqueue(target_z[:batch_size])

        # Returns q1 (view 1), q2 (view 2), nn1 (from view 1), nn2 (from view 2)
        #TODO: CHECK CONCAT?
        return q[:batch_size], q[batch_size:], nn1_target, nn2_target
