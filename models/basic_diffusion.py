from diffusers import UNet1DModel

class SimpleDiffuser(UNet1DModel):
    def __init__(self, inp_size=32, num_channels=32):
        super().__init__(
            sample_size=inp_size,
            in_channels=num_channels,
            out_channels=num_channels,
            layers_per_block=2,
            block_out_channels=(32, 64),
            down_block_types=("DownBlock1D", "AttnDownBlock1D"),
            up_block_types=("AttnUpBlock1D", "UpBlock1D"),
            use_timestep_embedding=True,
            act_fn="silu"
        )