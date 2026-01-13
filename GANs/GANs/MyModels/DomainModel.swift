//
//  DomainModel.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import Foundation
import MLX
import MLXNN

protocol DomainModel {
    static var latentDim: Int { get }
    static var url: URL { get }
    
    associatedtype Model: Module
    
    static func loadPretrained() throws -> Model
}
