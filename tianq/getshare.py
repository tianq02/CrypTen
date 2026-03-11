import torch
from numpy import dtype

import crypten
import crypten.mpc as mpc
import crypten.communicator as comm
import numpy as np

from crypten.mpc.primitives import ArithmeticSharedTensor

crypten.init()

def pt_przs(size, world_size=3):

    # shape: [..., world_size]
    # rnd = torch.randn(world_size, *size)
    from crypten.common.rng import generate_random_ring_element
    rnd = generate_random_ring_element([world_size, *size])

    # 在第一个维度向左轮换1位
    # 这样 rnd[i, ...] 对应 my_rnd
    #      roll_rnd[i, ...] 对应 received_rnd (来自上一方)
    roll_rnd = torch.roll(rnd, shifts=-1, dims=0)

    # 份额 = 我的随机数 - 收到的随机数
    # 对第一个维度进行逐元素相减
    pt_shares = rnd - roll_rnd

    return pt_shares

def pt_share(tensor: torch.Tensor, world_size=3):
    shares = pt_przs(tensor.size(), world_size)
    shares[0] += tensor * 65536  # 65536: simulate scaling in og cryptensor
    return shares

@mpc.run_multiprocess(world_size=2)
def test_load_share():
    rank = comm.get().get_rank()

    # 模拟从客户端接收的份额
    mock_shares = [
        torch.tensor([100, 200, 300]),  # S0 的份额
        torch.tensor([-100, -200, -300])  # S1 的份额 (互补)
    ]

    x = ArithmeticSharedTensor.from_shares(mock_shares[rank])
    print(f"Rank {rank}: share = {x.share}")

    # 解密验证
    revealed = x.reveal()
    print(f"Rank {rank}: revealed = {revealed}")  # 应该接近 [0, 0, 0]

if __name__ == '__main__':
    test_load_share()



@mpc.run_multiprocess(world_size=3)
def get_shares():
    # 2 steps are taken:
    # 1. all parties collaboratively generate PRZS (Pseudo-Random Zero Sharing).
    #    PRZS is designed such that sum of all shares equals zero.
    # 2. src party (default rank 0) adds plaintext to its PRZS share.
    #    the other party uses PRZS directly as its share.
    # so, you'll see 4 init messages for world_size=2,
    #    one for PRZS init, one for ASS tensor init, times 2 for 2 parties.
    # if you don't like those init messages, set DEBUG=False at arithmetic.py.

    # original crypten share
    x_enc: crypten.mpc.MPCTensor = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)

    rank = comm.get().get_rank()
    # print(f"\nrank: {rank}, share={x_enc.share}")

    # Return the raw underlying share back to the main process
    return rank, x_enc.share.numpy()

@mpc.run_multiprocess(world_size=3)
def get_share_outsourced():
    rank = comm.get().get_rank()


# test code for 2pc ASS
if __name__ == '__main__1':

    # The decorator returns a list: [result_from_rank_0, result_from_rank_1]
    shares = get_shares()

    # Now you have them as normal variables in Jupyter!
    for r, s in shares:
        print(f"Rank: {r}, Share: {s}")

    # You can even manually add them together to see the original fixed-point values (65536, 131072...)
    print("Sum & Scale:", sum(s for r, s in shares) / 65536)


# test code for outsourced ASS
if __name__ == '__main__2':

    shares = pt_przs([3])

    print(shares)
    print(shares[0])
    print(shares.sum(dim=0))

    arr = np.array([1,2,3],dtype=np.longlong)
    share2 = pt_share(torch.tensor(arr),3)

    print(share2)
    print(share2.sum(dim=0))