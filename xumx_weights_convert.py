import argparse
import torch
from asteroid.models.x_umx import XUMX
from model import X_UMX

parser = argparse.ArgumentParser()
parser.add_argument('infile', type=str)
parser.add_argument('outfile', type=str)

args = parser.parse_args()

orig = XUMX.from_pretrained(args.infile)
our = X_UMX(max_bins=1487)
with torch.no_grad():
    for i, src in enumerate(orig.sources):
        our.input_means[i*our.max_bins:(i+1) *
                        our.max_bins] = orig.mean_scale["input_mean_{}".format(src)]
        our.input_scale[i*our.max_bins:(
            i+1) * our.max_bins] = orig.mean_scale["input_scale_{}".format(src)]
        our.output_means[i*our.nb_output_bins:(
            i+1) * our.nb_output_bins] = orig.mean_scale["output_mean_{}".format(src)]
        our.output_scale[i*our.nb_output_bins:(
            i+1) * our.nb_output_bins] = orig.mean_scale["output_scale_{}".format(src)]

        getattr(our, f'{src}_lstm').load_state_dict(
            orig.layer_lstm[src].state_dict())

        our.affine1[0].weight[i*our.hidden_channels:(
            i+1) * our.hidden_channels, :, 0] = orig.layer_enc[src].enc[0].weight

        our.affine1[1].weight[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_enc[src].enc[1].weight
        our.affine1[1].bias[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_enc[src].enc[1].bias
        our.affine1[1].running_mean[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_enc[src].enc[1].running_mean
        our.affine1[1].running_var[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_enc[src].enc[1].running_var
        our.affine1[1].num_batches_tracked = orig.layer_enc[src].enc[1].num_batches_tracked

        our.affine2[0].weight[i*our.hidden_channels:(
            i+1) * our.hidden_channels, :, 0] = orig.layer_dec[src].dec[0].weight

        our.affine2[1].weight[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_dec[src].dec[1].weight
        our.affine2[1].bias[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_dec[src].dec[1].bias
        our.affine2[1].running_mean[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_dec[src].dec[1].running_mean
        our.affine2[1].running_var[i*our.hidden_channels:(
            i+1) * our.hidden_channels] = orig.layer_dec[src].dec[1].running_var
        our.affine2[1].num_batches_tracked = orig.layer_dec[src].dec[1].num_batches_tracked

        out_channels = our.nb_output_bins * our.nb_channels
        our.affine2[3].weight[i*out_channels:(i+1) * out_channels,
                              :, 0] = orig.layer_dec[src].dec[3].weight

        our.affine2[4].weight[i *
                              out_channels:(i+1) * out_channels] = orig.layer_dec[src].dec[4].weight
        our.affine2[4].bias[i *
                            out_channels:(i+1) * out_channels] = orig.layer_dec[src].dec[4].bias
        our.affine2[4].running_mean[i *
                                    out_channels:(i+1) * out_channels] = orig.layer_dec[src].dec[4].running_mean
        our.affine2[4].running_var[i *
                                   out_channels:(i+1) * out_channels] = orig.layer_dec[src].dec[4].running_var
        our.affine2[4].num_batches_tracked = orig.layer_dec[src].dec[4].num_batches_tracked

torch.save(our.state_dict(), args.outfile)
