//
//  VAE_Fashion_MNIST.swift
//  GANs
//
//  Created by Sharan Thakur on 09/12/25.
//

import Foundation
import MLX
import MLXNN

enum VAE_Fashion_MNIST: DomainModel {
    static let latentDim = 12
    static let url = Bundle.main.url(forResource: "decoder_weights_150", withExtension: "safetensors")!
    
    static func loadPretrained() throws -> Decoder {
        try Decoder.loadWeights(from: url)
    }
    
    class Decoder: Module, UnaryLayer {
        @ModuleInfo var fc: Linear
        @ModuleInfo var conv1: ConvTransposed2d
        @ModuleInfo var conv2: ConvTransposed2d
        @ModuleInfo var output: ConvTransposed2d
        
        override init() {
            self.fc = Linear(latentDim, 7 * 7 * 64)
            
            self.conv1 = ConvTransposed2d(
                inputChannels: 64,
                outputChannels: 64,
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
            self.conv2 = ConvTransposed2d(
                inputChannels: 64,
                outputChannels: 32,
                kernelSize: 3,
                stride: 2,
                padding: 1
            )
            self.output = ConvTransposed2d(
                inputChannels: 32,
                outputChannels: 1,
                kernelSize: 3,
                padding: 1
            )
        }
        
        func callAsFunction(_ x: MLXArray) -> MLXArray {
            let batchSize = x.shape[0]
            
            var o = relu(self.fc(x))
            o = reshaped(o, [batchSize, 7, 7, 64])
            
            o = relu(self.conv1(o))
            o = relu(self.conv2(o))
            o = sigmoid(self.output(o))
            
            return o
        }
        
        static func loadWeights(from url: URL) throws -> Decoder {
            let params = try loadSafetensors(from: url)
            
            let decoder = Decoder()
            try decoder.update(parameters: params, verify: .noUnusedKeys)
            
            eval(decoder.parameters())
            
            return decoder
        }
    }
}
