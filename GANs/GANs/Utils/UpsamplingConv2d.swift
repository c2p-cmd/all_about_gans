//
//  UpsamplingConv2d.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import MLX
import MLXNN

///
/// Upsampling followed by Conv2d layer
///
/// - Parameters:
///     - inputChannels: Number of input channels
///     - outputChannels: Number of output channels
///     - kernelSize: Size of the convolution kernel
///     - padding: Padding for the convolution
class UpsamplingConv2d: Module, UnaryLayer {
    @ModuleInfo var conv: Conv2d
    
    init(inputChannels: Int, outputChannels: Int, kernelSize: IntOrPair = 3, padding: IntOrPair = 1) {
        self.conv = Conv2d(
            inputChannels: inputChannels,
            outputChannels: outputChannels,
            kernelSize: kernelSize,
            stride: 1,
            padding: padding
        )
    }
    
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        self.conv(upsampleNearest(x: x))
    }
}

func upsampleNearest(x input: MLXArray, scale: Int = 2) -> MLXArray {
    var x = input
    assert(x.ndim == 4, "Input must be a 4D tensor [B, H, W, C]")
    let batchSize = x.shape[0]
    let height = x.shape[1]
    let width = x.shape[2]
    let channels = x.shape[3]
    
    // Insert singleton dimensions
    x = x.reshaped([batchSize, height, 1, width, 1, channels])
    // Broadcast to replicate values
    x = broadcast(x, to: [batchSize, height, scale, width, scale, channels])
    // Reshape to final output
    x = x.reshaped([batchSize, height * scale, width * scale, channels])
    return x
}
