from time import sleep

import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

crypten.init()

@mpc.run_multiprocess(world_size=2)
def get_shares():

    # 2 steps are taken:
    # 1. generate PRZS and share across both parties.
    # 2. rank1 add PRZS to plaintext to get its share.
    #    rank2 use PRZS as its share.
    # so, you'll see 4 init messages, one for PRZS init, one for ASS tensor init, times 2 for 2 parties.
    # if you don't like those messages, set DEBUG=False at arithmetic.py.
    x_enc: crypten.mpc.MPCTensor = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)

    rank = comm.get().get_rank()

    # avoid messing up outputs
    # sleep(rank/100)

    # barrier: both code waits at barrier to sync, making the output always stable
    # here, rank 0 print and wait for rank 1
    # rank 1 wait for rank 0 done printing, then print its output
    # and yes, syncing makes performance overhead
    if rank == 0:
        print(f"rank: {rank}, share={x_enc.share}")
        comm.get().barrier()
    else:
        comm.get().barrier()
        print(f"rank: {rank}, share={x_enc.share}")


    # Return the raw underlying share back to the main process
    return x_enc.share.numpy()

# The decorator returns a list: [result_from_rank_0, result_from_rank_1]
shares = get_shares()

# Now you have them as normal variables in Jupyter!
share_0 = shares[0]
share_1 = shares[1]

print("Share 0:", share_0)
print("Share 1:", share_1)

# You can even manually add them together to see the original fixed-point values (65536, 131072...)
print("Sum of shares:", share_0 + share_1)