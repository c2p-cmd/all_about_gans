//
//  ViewModifiers.swift
//  GANs
//
//  Created by Sharan Thakur on 09/12/25.
//

import SwiftUI

extension View {
    @ViewBuilder
    func buttonSizeIfAvailable(flexible: Bool) -> some View {
        if #available(iOS 26, macOS 26, *) {
            self.buttonSizing(flexible ? .flexible : .fitted)
        } else {
            self
        }
    }
}
