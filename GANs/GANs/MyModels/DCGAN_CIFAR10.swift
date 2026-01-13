//
//  DCGAN_CIFAR10.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import Foundation
import MLX
import MLXNN

enum DCGANCIFAR10: DomainModel {
    static let latentDim = 100
    static let url = Bundle.main.url(forResource: "generator_epoch_20", withExtension: "safetensors")!
    
    static func loadPretrained() throws -> Generator {
        return try Generator.loadWeights(from: url)
    }
    
    class Generator: Module, UnaryLayer {
        @ModuleInfo var linear: Linear
        @ModuleInfo var bn0: BatchNorm
        
        @ModuleInfo var up1: UpsamplingConv2d
        @ModuleInfo var bn1: BatchNorm
        
        @ModuleInfo var up2: UpsamplingConv2d
        @ModuleInfo var bn2: BatchNorm
        
        @ModuleInfo var up3: UpsamplingConv2d
        
        init(latentDim: Int = 100) {
            self.linear = Linear(latentDim, 4 * 4 * 256)
            self.bn0 = BatchNorm(featureCount: 4 * 4 * 256)
            
            self.up1 = UpsamplingConv2d(
                inputChannels: 256,
                outputChannels: 128,
            )
            self.bn1 = BatchNorm(featureCount: 128)
            
            self.up2 = UpsamplingConv2d(
                inputChannels: 128,
                outputChannels: 64,
            )
            self.bn2 = BatchNorm(featureCount: 64)
            
            self.up3 = UpsamplingConv2d(
                inputChannels: 64,
                outputChannels: 3,
            )
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var o = relu(bn0(linear(x)))
            o = reshaped(o, [-1, 4, 4, 256])
            
            o = relu(bn1(up1(o)))
            o = relu(bn2(up2(o)))
            
            return tanh(up3(o))
        }
        
        static func loadWeights(from url: URL) throws -> Generator {
            let params = try loadSafetensorsVAE(from: url)
            
            let generator = Generator(latentDim: latentDim)
            generator.apply(filter: isConvWeight(_:key:value:), map: initConvWeight(_:))
            generator.apply(filter: isBatchNormWeight(_:key:value:), map: initBatchNormWeight(_:))
            
            try generator.update(parameters: params, verify: .all)
            
            eval(generator.parameters())
            
            return generator
        }
    }
}
