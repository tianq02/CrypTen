import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

crypten.init()

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
    x_enc: crypten.mpc.MPCTensor = crypten.cryptensor([1.0, 2.0, 3.0], ptype=crypten.mpc.arithmetic)

    rank = comm.get().get_rank()

    # print(f"\nrank: {rank}, share={x_enc.share}")

    # Return the raw underlying share back to the main process
    return x_enc.share.numpy()

# The decorator returns a list: [result_from_rank_0, result_from_rank_1]
shares = get_shares()

# Now you have them as normal variables in Jupyter!
for i,s in enumerate(shares):
    print(f"Share {i}:{s}")

# You can even manually add them together to see the original fixed-point values (65536, 131072...)
print("Sum & Scale:", sum(shares)/65536)