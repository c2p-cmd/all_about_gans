//
//  MLXArray+.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import SwiftUI
import MLX
import MLXNN

func clipArray(x: MLXArray) -> MLXArray {
    clip(x, min: 0.0, max: 255.0)
}

extension MLXArray {
    /// Converts a grayscale MLXArray (H, W) or (H, W, 1) with values in [-1, 1] to a NativeImage
    func grayscaleToNativeImage(denormalize: (MLXArray) -> MLXArray) -> NativeImage? {
        // 1. Denormalize
        var normalized = denormalize(self)
        
        // 2. Cast to UInt8 (0-255 integers)
        normalized = normalized.asType(.uint8)
        
        // 3. Ensure flattened data extraction
        let bytes = normalized.asArray(UInt8.self)
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
    
    
    func rgbToNativeImage(denormalize: (MLXArray) -> MLXArray) -> NativeImage? {
        // 1. Check Dimensions (Safety Check)
        // We expect a single image here, effectively 3 dimensions.
        // If this is 4D (Batch, C, H, W), the user needs to slice it first.
        guard self.ndim == 3 else {
            print("Error: rgbToNativeImage expects a 3D array (C, H, W) or (H, W, C). Received \(self.ndim)D.")
            return nil
        }
        
        // 2. Handle Layout: Convert (C, H, W) -> (H, W, C)
        // If the first dimension is 3 (RGB), it's likely Channel-First.
        // We need Channel-Last for CGImage.
        var imageTensor = self
        if imageTensor.dim(0) == 3 {
            imageTensor = imageTensor.transposed(axes: [1, 2, 0]) // Permute dimensions
        }
        
        // 3. Denormalize
        var normalized = denormalize(imageTensor)
        
        // 4. Clip and Cast
        // strictly clip 0...255 to prevent integer wrap-around artifacts
        normalized = clipArray(x: normalized).asType(.uint8)
        
        // 5. Get Dimensions
        let height = normalized.dim(0)
        let width = normalized.dim(1)
        let channels = normalized.dim(2) // Should be 3
        
        // 6. Flatten data
        let bytes = normalized.asArray(UInt8.self)
        
        // 7. Create CGImage
        let bitsPerComponent = 8
        let bitsPerPixel = bitsPerComponent * channels
        let bytesPerRow = width * channels
        let colorSpace = CGColorSpaceCreateDeviceRGB()
        // Use explicit bitmap info to ensure RGB interpretation
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
            shouldInterpolate: true, // True usually looks better for generated images
            intent: .defaultIntent
        ) else { return nil }
        
        // 8. Platform Return
#if os(macOS)
        return NSImage(cgImage: cgImage, size: NSSize(width: width, height: height))
#elseif os(iOS)
        return UIImage(cgImage: cgImage)
#endif
    }
    
}
