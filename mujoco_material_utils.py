import numpy as np

# ==========================================
# Class: FlexPlateLayer
# Single Layer Definition with Material Properties
# ==========================================
class FlexPlateLayer:
    def __init__(self, name, width, depth, thickness, mass, 
                 youngs_modulus=1e9, damping_ratio=0.01, 
                 rgba=None, material_model='linear', prony_series=None):
        self.name = name
        self.width = width
        self.depth = depth
        self.thickness = thickness
        self.mass = mass
        
        # Material Properties
        self.E = youngs_modulus
        self.damping_ratio = damping_ratio
        
        # Material Model: 'linear', 'neo_hookean' (Hyperelastic), 'hyperfoam'
        self.material_model = material_model
        
        # Viscoelasticity (Prony Series)
        # Format: [(E_i, tau_i), ...] for relaxation terms
        self.prony_series = prony_series 
        
        self.rgba = rgba if rgba is not None else [0.5, 0.5, 0.5, 1.0]

    def get_xml(self, uid, offset_z):
        # Resolution: Simplified to 2x2 for speed optimization as requested
        res_x = 2
        res_y = 2
        nc = f"{res_x} {res_y} 2"
        
        # --- Parameter Conversion ---
        # 1. Stiffness (approximate for 'box' type flex)
        # MuJoCo 'stiffness' is not exactly Young's Modulus. 
        # Typically requires scaling factor based on resolution/size.
        # Simple heuristic: k ~ E * scale
        scale_factor = 1.0 # TBD via Calibration
        k_val = self.E * scale_factor
        
        # 2. Damping
        d_val = self.damping_ratio * 100.0 # Heuristic scaling
        
        # 3. Solimp/Solref for Material Models
        solref_str = "0.02 1" # Default
        solimp_str = "0.9 0.95 0.001 0.5 2" # Default Linear-ish
        
        if self.material_model == 'hyperfoam':
            # Nonlinear Force Profile: Low width (soft start), High power (hardening)
            solimp_str = "0.001 0.99 0.01 0.5 6" 
            # Solref: overdamped for foam
            solref_str = "0.01 2" 
            
        elif self.material_model == 'neo_hookean':
            # Neo-Hookean mimics rubber
            # Higher Poisson ratio effect via solimp? Not directly.
            # Using standard solimp but lower solref time constant
            solref_str = "0.005 1"
        
        # 4. Viscoelasticity (Prony Series Approximated via Solref/Damping)
        if self.prony_series:
            # MuJoCo 3.0 basic flex doesn't take Prony coefficients directly.
            # We approximate by increasing damping and solref time constant
            # derived from average tau of the series.
            avg_tau = sum([p[1] for p in self.prony_series]) / len(self.prony_series)
            # Time const in Solref
            solref_str = f"{avg_tau:.4f} 2" # Critical damping with relaxation time
            
            # Increase damping based on relaxation strength
            relax_E = sum([p[0] for p in self.prony_series])
            d_val += (relax_E / self.E) * 50.0

        # Calculate Spacing for FlexComp (MuJoCo 3.4+ requirement)
        # Count is res_x, res_y, 2
        # Spacing = Dimension / (Count - 1)
        sx = self.width / max(res_x - 1, 1)
        sy = self.depth / max(res_y - 1, 1)
        sz = self.thickness / max(2 - 1, 1) # Thickness has 2 layers -> spacing = thickness
        spacing_str = f"{sx:.6f} {sy:.6f} {sz:.6f}"

        return f"""
        <body name="{self.name}_{uid}" pos="0 0 {offset_z}">
            <!-- Flex Body: Native Elasticity -->
            <flexcomp name="flex_{self.name}_{uid}" type="grid" dim="3" spacing="{spacing_str}"
                      count="{nc}" mass="{self.mass}"
                      radius="0.0005"
                      rgba="{self.rgba[0]} {self.rgba[1]} {self.rgba[2]} {self.rgba[3]}">
                <elasticity young="{self.E}" poisson="{'0.45' if self.material_model == 'hyperfoam' else '0.3'}" damping="{self.damping_ratio}"/>
                <contact selfcollide="none"/>
            </flexcomp>
            
            <!-- Sites for 'Edge/Corner' Connectivity -->
            <!-- Corners: SW, NW, SE, NE -->
            <site name="s_{self.name}_{uid}_c0" pos="{-self.width/2} {-self.depth/2} 0" size="0.01" rgba="1 0 0 0.5"/>
            <site name="s_{self.name}_{uid}_c1" pos="{-self.width/2} {self.depth/2} 0" size="0.01" rgba="1 0 0 0.5"/>
            <site name="s_{self.name}_{uid}_c2" pos="{self.width/2} {-self.depth/2} 0" size="0.01" rgba="1 0 0 0.5"/>
            <site name="s_{self.name}_{uid}_c3" pos="{self.width/2} {self.depth/2} 0" size="0.01" rgba="1 0 0 0.5"/>
        </body>
        """

# ==========================================
# Class: MultiLayerPanel
# Assembly of FlexPlateLayers with Connections
# ==========================================
class MultiLayerPanel:
    def __init__(self, layers, connection_type='full'):
        self.layers = layers # List of FlexPlateLayer
        self.connection_type = connection_type # 'full', 'edge', 'corner'
        
    def get_xml(self, uid, start_z):
        # Calculate Z centering
        total_thk = sum([l.thickness for l in self.layers])
        z_cursor = -total_thk / 2.0
        
        bodies_xml = ""
        connections_xml = ""
        
        prev_layer_name = None
        
        for i, layer in enumerate(self.layers):
            center_z = z_cursor + layer.thickness / 2.0
            bodies_xml += layer.get_xml(uid, center_z)
            
            curr_layer_name = f"{layer.name}_{uid}"
            
            if prev_layer_name:
                # Inter-Layer Connections
                if self.connection_type == 'full':
                    # Weld Bodies (Rigid Frame Lock)
                    connections_xml += f"""
                    <equality>
                        <weld name="weld_{prev_layer_name}_{curr_layer_name}" 
                              body1="{prev_layer_name}" body2="{curr_layer_name}"
                              solref="0.01 1" solimp="0.95 0.99 0.001 0.5 4"/>
                    </equality>
                    """
                elif self.connection_type == 'corner':
                    # Connect 4 Corners via Sites
                    for ci in range(4):
                        connections_xml += f"""
                        <equality>
                            <connect name="conn_{prev_layer_name}_{curr_layer_name}_c{ci}"
                                     body1="{prev_layer_name}" body2="{curr_layer_name}"
                                     anchor1="0 0 0" 
                                     site1="s_{prev_layer_name}_c{ci}" site2="s_{curr_layer_name}_c{ci}"
                                     solref="0.02 1"/>
                        </equality>
                        """
                elif self.connection_type == 'edge':
                    # Connect Corners (simplified edge)
                    # For full edge, we need more sites along the edge.
                    # Fallback to corner connect for now
                    for ci in range(4):
                        connections_xml += f"""
                        <equality>
                            <connect name="conn_{prev_layer_name}_{curr_layer_name}_c{ci}"
                                     body1="{prev_layer_name}" body2="{curr_layer_name}"
                                     site1="s_{prev_layer_name}_c{ci}" site2="s_{curr_layer_name}_c{ci}"
                                     solref="0.02 1"/>
                        </equality>
                        """

            prev_layer_name = curr_layer_name
            z_cursor += layer.thickness

        return bodies_xml, connections_xml

# ==========================================
# Utility: Material Curve Fitter
# ==========================================
class MaterialFitter:
    def __init__(self):
        pass
        
    def fit_material(self, target_data, spec_size):
        """
        Find optimal MuJoCo params (stiffness, damping, solimp) to match target curve.
        target_data: List of (strain, stress) or (displacement, force) tuples
        spec_size: (L, W, H) of specimen
        """
        print("🧪 Running Material Calibration (Placeholder)...")
        # In a real implementation, this would:
        # 1. Build a simple 1-element simulation scene (Compression/Bending)
        # 2. Run simulation with current guess params
        # 3. Compute error (MSE)
        # 4. Optimize using scipy.minimize or gradient descent
        
        # For now, return a heuristic based on linear fit of the target data
        x, y = zip(*target_data)
        slope = (y[-1] - y[0]) / (x[-1] - x[0]) if len(x) > 1 else 1.0
        
        print(f"   -> Estimated E-Modulus from slope: {slope:.2e}")
        return {
            'youngs_modulus': slope,
            'damping_ratio': 0.05,
            'material_model': 'linear'
        }
