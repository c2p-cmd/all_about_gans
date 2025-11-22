//
//  Utils.swift
//  GANs
//
//  Created by Sharan Thakur on 13/11/25.
//

import Foundation
import MLX
import MLXNN

func loadSafetensors(from url: URL) throws -> ModuleParameters {
    let weights: [String: MLXArray] = try loadArrays(url: url)
    
    var nestedParams = [String: NestedItem<String, MLXArray>]()
    
    let groupedByLayer: [String : [Dictionary<String, MLXArray>.Element]] = Dictionary(grouping: weights) { key, _ in
        key.split(separator: ".").first.map(String.init) ?? ""
    }
    
    for (layerName, params) in groupedByLayer {
        if layerName.isEmpty {
            print("Warning: Encountered parameter with empty layer name. Skipping.")
            continue
        }
        
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
    return ModuleParameters(values: nestedParams)
}

func loadSafetensorsVAE(from url: URL) throws -> ModuleParameters {
    let weights: [String: MLXArray] = try loadArrays(url: url)
        
    // Start with an empty root dictionary
    var rootParams = [String: NestedItem<String, MLXArray>]()
    
    for (fullKey, array) in weights {
        // Split "decoder.bn1.bias" into ["decoder", "bn1", "bias"]
        let keyPath = fullKey.split(separator: ".").map(String.init)
        
        // Insert deeply into the dictionary
        recursiveInsert(keys: keyPath, value: array, into: &rootParams)
    }
    
    return ModuleParameters(values: rootParams)
}

/// Helper function to recursively build the NestedItem hierarchy
fileprivate func recursiveInsert(keys: [String], value: MLXArray, into dict: inout [String: NestedItem<String, MLXArray>]) {
    guard !keys.isEmpty else {
        print("Problem with key extraction, bailing out.")
        return
    }
    
    let head = keys[0]
    let tail = Array(keys.dropFirst())
    
    if tail.isEmpty {
        // Base Case: We are at the end of the path (e.g., "bias"), insert the Value
        dict[head] = .value(value)
    } else {
        // Recursive Case: We are at a node (e.g., "decoder" or "bn1")
        
        // 1. Get the existing dictionary for this key, or create a new empty one
        var subDict: [String: NestedItem<String, MLXArray>] = [:]
        
        if let existingItem = dict[head], case .dictionary(let d) = existingItem {
            subDict = d
        }
        
        // 2. Recurse down the path
        recursiveInsert(keys: tail, value: value, into: &subDict)
        
        // 3. Update the current dictionary with the modified child
        dict[head] = .dictionary(subDict)
    }
}

func generateLatentVectors(batchSize: Int = 16, latentDim: Int = 100) -> MLXArray {
    let shape = [batchSize, latentDim]
    return MLXRandom.normal(shape)
}

func denormalizeTanH(_ x: MLXArray) -> MLXArray {
    // Normalize from [-1, 1] to [0, 255.0]
    return (x + 1.0) * 127.5
}

func denormalizeSigmoid(_ x: MLXArray) -> MLXArray {
    // Normalize from [0, 1] to [0, 255.0]
    return x * 255.0
}

func initConvWeight(_ x: MLXArray) -> MLXArray {
    MLXRandom.normal(loc: 0.0, scale: 0.02)
}
func initBatchNormWeight(_ x: MLXArray) -> MLXArray {
    MLXRandom.normal(loc: 1.0, scale: 0.02)
}

func isConvWeight(_ module: Module, key: String, value: ModuleItem) -> Bool {
    return (module is Conv2d || module is ConvTransposed2d) && key.contains("weight")
}

func isBatchNormWeight(_ module: Module, key: String, value: ModuleItem) -> Bool {
    return (module is BatchNorm) && (key.contains("weight"))
}
