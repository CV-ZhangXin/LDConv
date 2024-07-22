# the different initial sampled shapes (a) (b) (c) (d) (e)
# we have explored the size 5 LDConv.
def _get_p_n1(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 1, 2, 2])
        p_n_y = torch.tensor([0, 2, 1, 0, 2])
        p_n = torch.cat([p_n_x,p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_n2(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 0, 1, 1])
        p_n_y = torch.tensor([0, 1, 2, 0, 2])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_n3(self, N, dtype):
        p_n_x = torch.tensor([0, 0, 0, 0, 1])
        p_n_y = torch.tensor([0, 1, 2, 3, 2])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_n4(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 2, 3, 4])
        p_n_y = torch.tensor([0, 1, 2, 3, 4])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n

    def _get_p_n5(self, N, dtype):
        p_n_x = torch.tensor([0, 1, 1, 1, 2])
        p_n_y = torch.tensor([0, 1, 2, 3, 0])
        p_n = torch.cat([p_n_x, p_n_y], 0)
        p_n = p_n.view(1, 2 * N, 1, 1).type(dtype)
        return p_n
