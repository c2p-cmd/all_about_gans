//
//  NativeImage.swift
//  GANs
//
//  Created by Sharan Thakur on 21/11/25.
//

import SwiftUI
#if canImport(AppKit)
typealias NativeImage = NSImage
#elseif canImport(UIKit)
typealias NativeImage = UIImage
#endif
