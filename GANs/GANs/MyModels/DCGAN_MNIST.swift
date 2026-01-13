//
//  DCGAN_MNIST.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import Foundation
import MLX
import MLXNN

enum DCGANMNIST: DomainModel {
    static let latentDim = 100
    static let url = Bundle.main.url(forResource: "dcgan_generator_weights", withExtension: "safetensors")!
    
    static func loadPretrained() throws -> Generator {
        return try Generator.loadWeights(from: url)
    }
    
    public class Generator: Module, UnaryLayer {
        @ModuleInfo var fc: Linear
        @ModuleInfo var conv1: ConvTransposed2d
        @ModuleInfo var conv2: ConvTransposed2d
        @ModuleInfo var conv3: ConvTransposed2d
        @ModuleInfo var bn1: BatchNorm
        @ModuleInfo var bn2: BatchNorm
        
        init(latentDim: Int = 100) {
            self.fc = Linear(latentDim, 7 * 7 * 256)
            
            self.conv1 = ConvTransposed2d(
                inputChannels: 256,
                outputChannels: 128,
                kernelSize: 5,
                stride: 1,
                padding: 2
            )
            self.bn1 = BatchNorm(featureCount: 128)
            
            self.conv2 = ConvTransposed2d(
                inputChannels: 128,
                outputChannels: 64,
                kernelSize: 5,
                stride: 2,
                padding: 2
            )
            self.bn2 = BatchNorm(featureCount: 64)
            
            self.conv3 = ConvTransposed2d(
                inputChannels: 64,
                outputChannels: 1,
                kernelSize: 5,
                stride: 2,
                padding: 2
            )
        }
        
        public func callAsFunction(_ z: MLX.MLXArray) -> MLX.MLXArray {
            var x = fc(z)
            x = reshaped(x, [x.shape[0], 7, 7, 256])
            x = relu(bn1(conv1(x)))
            x = relu(bn2(conv2(x)))
            x = tanh(conv3(x))
            return x
        }
        
        static func loadWeights(from url: URL) throws -> Generator {
            let params = try loadSafetensors(from: url)
            
            let generator = Generator()
            generator.apply(filter: isConvWeight(_:key:value:), map: initConvWeight(_:))
            generator.apply(filter: isBatchNormWeight(_:key:value:), map: initBatchNormWeight(_:))
            try generator.update(parameters: params, verify: .all)
            
            eval(generator.parameters())
            
            return generator
        }
    }
}
