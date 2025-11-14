//
//  Utils.swift
//  GANs
//
//  Created by Sharan Thakur on 13/11/25.
//

import SwiftUI
import MLX
import MLXNN

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
        let weights = try loadArrays(url: url)
        
        var nestedParams = [String: NestedItem<String, MLXArray>]()

        let groupedByLayer = Dictionary(grouping: weights) { key, _ in
            key.split(separator: ".").first.map(String.init) ?? ""
        }
        
        for (layerName, params) in groupedByLayer {
            if layerName.isEmpty { continue }
            
            var layerDict = [String: NestedItem<String, MLXArray>]()
            for (fullKey, value) in params {
                // Get the param name (e.g., "weight", "bias")
                let paramName = fullKey.split(separator: ".").dropFirst().joined(separator: ".")
                if !paramName.isEmpty {
                    layerDict[paramName] = .value(value)
                }
            }
            
            nestedParams[layerName] = .dictionary(layerDict)
        }

        let generator = Generator()
        
        try generator.update(parameters: ModuleParameters(values: nestedParams), verify: .all)
        
        eval(generator.parameters())
        
        return generator
    }
}

func generateLatentVectors(batchSize: Int = 16, latentDim: Int = 100) -> MLX.MLXArray {
    let shape = [batchSize, latentDim]
    return MLXRandom.normal(shape)
}

extension MLXArray {
    /// Converts a grayscale MLXArray (H, W) or (H, W, 1) with values in [-1, 1] to a NativeImage
    func toNativeImage() -> NativeImage? {
        // 1. Denormalize: Convert [-1, 1] -> [0, 255]
        // formula: (x + 1) / 2 * 255
        var normalized = (self + 1.0) * 127.5
        
        // 2. Cast to UInt8 (0-255 integers)
        normalized = normalized.asType(.uint8)
        
        // 3. Ensure CPU sync and flattened data extraction
        // Note: asArray(Type) returns a flattened array
        let bytes = normalized.asArray(UInt8.self)
        
        // Get dimensions
        let height = self.dim(0)
        let width = self.dim(1)
        
        // 4. Create CGImage
        // Grayscale parameters
        let bitsPerComponent = 8
        let bitsPerPixel = 8
        let bytesPerRow = width // 1 byte per pixel for grayscale
        let colorSpace = CGColorSpaceCreateDeviceGray()
        let bitmapInfo = CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue)
        
        guard let provider = CGDataProvider(data: Data(bytes) as CFData) else { return nil }
        
        guard let cgImage = CGImage(
            width: width,
            height: height,
            bitsPerComponent: bitsPerComponent,
            bitsPerPixel: bitsPerPixel,
            bytesPerRow: bytesPerRow,
            space: colorSpace,
            bitmapInfo: bitmapInfo,
            provider: provider,
            decode: nil,
            shouldInterpolate: false,
            intent: .defaultIntent
        ) else { return nil }
        
        // 5. Convert to Platform specific Image
        #if os(macOS)
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
        #elseif os(iOS)
        return UIImage(cgImage: cgImage)
        #endif
    }
}
