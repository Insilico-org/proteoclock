"""
Factory module for proteomic aging clock implementations.

This module provides a factory method for creating aging clock objects
from the proteoclock library, supporting different types of clocks
and configurations.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal, Any
import pandas as pd
import pkg_resources
from collections import defaultdict

from .clocks.simple_clocks.clocks import (
    BaseAgingClock, GompertzClock, LinearClock, CPHClock
)

class ClockFactory:
    """Factory for creating aging clock instances with automatic resource discovery."""
    
    def __init__(self):
        """Initialize the factory and discover available resources."""
        self.materials_path = self._find_materials_path()
        self.simple_clocks_path = self.materials_path / "simple_clocks"
        self.deep_clocks_path = self.materials_path / "deep_clocks"
        self.scalers_path = self.materials_path / "scalers"
        
        # Cache for discovered clocks
        self._clock_cache = None
        self._scaler_cache = None
    
    def _find_materials_path(self) -> Path:
        """Find the materials directory using multiple strategies."""
        # Strategy 1: Try pkg_resources (works for installed packages)
        try:
            resource_path = pkg_resources.resource_filename('proteoclock', 'materials')
            if os.path.exists(resource_path):
                return Path(resource_path)
        except:
            pass
        
        # Strategy 2: Relative to this file (works in development)
        current_file = Path(__file__).resolve()
        materials_path = current_file.parent / "materials"
        if materials_path.exists():
            return materials_path
        
        # Strategy 3: Walk up the directory tree looking for proteoclock/materials
        current = current_file.parent
        for _ in range(5):  # Limit search depth
            potential_path = current / "proteoclock" / "materials"
            if potential_path.exists():
                return potential_path
            current = current.parent
        
        # Strategy 4: Check if we're already in proteoclock directory
        if current_file.parent.name == "proteoclock":
            materials_path = current_file.parent / "materials"
            if materials_path.exists():
                return materials_path
        
        raise FileNotFoundError(
            f"Could not locate materials directory. Searched from: {current_file}"
        )
    
    def _discover_clocks(self) -> Dict[str, Dict]:
        """Discover all available clocks and their configurations."""
        if self._clock_cache is not None:
            return self._clock_cache
        
        clocks = {}
        
        # Discover simple clocks
        if self.simple_clocks_path.exists():
            for clock_dir in self.simple_clocks_path.iterdir():
                if clock_dir.is_dir():
                    clock_name = clock_dir.name
                    weights_dir = clock_dir / "weights"
                    
                    if weights_dir.exists():
                        clocks[clock_name] = {
                            'type': 'simple',
                            'path': clock_dir,
                            'variants': self._discover_simple_clock_variants(weights_dir)
                        }
        
        # Discover deep clocks
        if self.deep_clocks_path.exists():
            for clock_dir in self.deep_clocks_path.iterdir():
                if clock_dir.is_dir():
                    clock_name = clock_dir.name
                    # Check for required deep clock files
                    if (clock_dir / "best_clock.pt").exists() and (clock_dir / "feature_order.txt").exists():
                        clocks[clock_name] = {
                            'type': 'deep',
                            'path': clock_dir,
                            'variants': {'default': str(clock_dir)}
                        }
        
        self._clock_cache = clocks
        return clocks
    
    def _discover_simple_clock_variants(self, weights_dir: Path) -> Dict[str, Dict]:
        """Discover variants for a simple clock."""
        variants = {}
        
        if weights_dir.name == "weights" and not list(weights_dir.glob("*.txt")):
            # Nested structure (like goeminne_2025)
            for subtype_dir in weights_dir.iterdir():
                if subtype_dir.is_dir():
                    subtype_name = subtype_dir.name
                    weight_files = list(subtype_dir.glob("*.txt"))
                    
                    # Parse weight files to find conventional vs organ-specific
                    conventional = []
                    organs = []
                    
                    for weight_file in weight_files:
                        if "conventional" in weight_file.name:
                            conventional.append(weight_file.name)
                        else:
                            # Extract organ name from filename
                            # Assuming format like "gladyshev_intestine_1st_gen_r.txt"
                            parts = weight_file.stem.split('_')
                            if len(parts) >= 2:
                                organ = parts[1]  # Second part is usually the organ
                                organs.append(organ)
                    
                    variants[subtype_name] = {
                        'path': subtype_dir,
                        'conventional': len(conventional),
                        'organs': sorted(set(organs)),
                        'files': [f.name for f in weight_files]
                    }
        else:
            # Flat structure (like kuo_2024)
            weight_files = list(weights_dir.glob("*.txt"))
            variants['default'] = {
                'path': weights_dir,
                'conventional': len(weight_files),
                'organs': [],
                'files': [f.name for f in weight_files]
            }
        
        return variants
    
    def _discover_scalers(self) -> Dict[str, Path]:
        """Discover available scalers."""
        if self._scaler_cache is not None:
            return self._scaler_cache
        
        scalers = {}
        if self.scalers_path.exists():
            for scaler_file in self.scalers_path.glob("*.pckl"):
                scaler_name = scaler_file.stem
                scalers[scaler_name] = scaler_file
        
        self._scaler_cache = scalers
        return scalers
    
    def view_clocks(self) -> None:
        """Display all available clocks in a user-friendly format."""
        clocks = self._discover_clocks()
        
        if not clocks:
            print("No clocks found.")
            return
        
        print("Available Aging Clocks:")
        print("=" * 50)
        
        # Group by type
        simple_clocks = {k: v for k, v in clocks.items() if v['type'] == 'simple'}
        deep_clocks = {k: v for k, v in clocks.items() if v['type'] == 'deep'}
        
        if simple_clocks:
            print("\nSimple Clocks:")
            print("-" * 20)
            for clock_name, clock_info in simple_clocks.items():
                print(f"  {clock_name}:")
                for variant_name, variant_info in clock_info['variants'].items():
                    conventional_count = variant_info['conventional']
                    organ_count = len(variant_info['organs'])
                    if variant_name == 'default':
                        print(f"    - {conventional_count} conventional clock(s)")
                    else:
                        print(f"    {variant_name}:")
                        print(f"      - {conventional_count} conventional, {organ_count} organ-specific")
                        if organ_count > 0:
                            organs_str = ", ".join(variant_info['organs'][:5])
                            if organ_count > 5:
                                organs_str += f", ... ({organ_count-5} more)"
                            print(f"      - Organs: {organs_str}")
        
        if deep_clocks:
            print("\nDeep Learning Clocks:")
            print("-" * 25)
            for clock_name in deep_clocks:
                print(f"  {clock_name}")
    
    def view_clock_variants(self, clock_name: str) -> None:
        """Display detailed variants for a specific clock."""
        clocks = self._discover_clocks()
        
        if clock_name not in clocks:
            available_clocks = list(clocks.keys())
            print(f"Clock '{clock_name}' not found.")
            print(f"Available clocks: {', '.join(available_clocks)}")
            return
        
        clock_info = clocks[clock_name]
        print(f"Variants for {clock_name}:")
        print("=" * (len(clock_name) + 13))
        
        if clock_info['type'] == 'simple':
            for variant_name, variant_info in clock_info['variants'].items():
                if variant_name == 'default':
                    print(f"\nDefault variant:")
                else:
                    print(f"\n{variant_name}:")
                
                print(f"  Conventional clocks: {variant_info['conventional']}")
                if variant_info['organs']:
                    print(f"  Organ-specific clocks: {len(variant_info['organs'])}")
                    print(f"  Available organs: {', '.join(variant_info['organs'])}")
                
                print(f"  Weight files: {len(variant_info['files'])}")
                for file in variant_info['files'][:3]:  # Show first 3 files
                    print(f"    - {file}")
                if len(variant_info['files']) > 3:
                    print(f"    ... and {len(variant_info['files'])-3} more")
        
        elif clock_info['type'] == 'deep':
            print("\nDeep learning clock - single variant available")
            print("  Files: best_clock.pt, feature_order.txt")
    
    def view_clock_scalers(self, clock_name: str = None) -> None:
        """Display available scalers, optionally filtered by clock compatibility."""
        scalers = self._discover_scalers()
        
        if not scalers:
            print("No scalers found.")
            return
        
        print("Available Scalers:")
        print("=" * 20)
        
        if clock_name:
            # Filter scalers that might be compatible with the clock
            compatible_scalers = {}
            for scaler_name, scaler_path in scalers.items():
                if any(part in scaler_name for part in clock_name.split('_')):
                    compatible_scalers[scaler_name] = scaler_path
            
            if compatible_scalers:
                print(f"\nSuggested scalers for {clock_name}:")
                for scaler_name in compatible_scalers:
                    print(f"  - {scaler_name}")
            
            print(f"\nAll available scalers:")
        
        for scaler_name in scalers:
            print(f"  - {scaler_name}")
    
    def get_clock(self,
                  name: str,
                  subtype: str = "conventional",
                  scaler: Optional[str] = None) -> BaseAgingClock:
        """
        Create and return a clock instance.

        Args:
            name: Clock name. Can be base name (e.g., 'kuo_2024', 'goeminne_2025') 
                  or extended with variant (e.g., 'goeminne_2025_full_chrono')
            subtype: For simple clocks, either 'conventional' or organ name
            scaler: Optional scaler name

        Returns:
            Configured aging clock instance
        """
        # Parse the clock name to extract base name and variant
        base_name, variant = self._parse_clock_name(name)
        
        clocks = self._discover_clocks()

        if base_name not in clocks:
            available_clocks = self._get_all_available_clock_names()
            raise ValueError(f"Clock '{name}' not found. Available: {available_clocks}")

        clock_info = clocks[base_name]
        
        # Handle deep clocks
        if clock_info['type'] == 'deep':
            if name == 'galkin_2025':
                from .clocks.deep_clocks.nn_wrapper import AgingClockPredictor
                weights_path = clock_info['path'] / "best_clock.pt"
                features_path = clock_info['path'] / "feature_order.txt"
                return AgingClockPredictor(str(weights_path), str(features_path))
            else:
                raise NotImplementedError(f"Deep clock '{name}' not yet supported")
        
        # Handle simple clocks
        elif clock_info['type'] == 'simple':
            # Determine the correct variant and weight file
            coef_file = self._find_weight_file(clock_info, variant, subtype)
            scaler_file = self._find_scaler_file(scaler) if scaler else None
            
            # Create appropriate clock type based on name and variant
            if name.startswith('kuo'):
                return GompertzClock(coef_file=str(coef_file), scaler_file=str(scaler_file) if scaler_file else None)
            elif name.startswith('goeminne'):
                # Determine clock type based on variant
                effective_variant = variant or list(clock_info['variants'].keys())[0]
                if 'chrono' in effective_variant:
                    return LinearClock(coef_file=str(coef_file), scaler_file=str(scaler_file) if scaler_file else None)
                elif 'mortality' in effective_variant:
                    return CPHClock(coef_file=str(coef_file), scaler_file=str(scaler_file) if scaler_file else None)
                else:
                    # Default to LinearClock
                    return LinearClock(coef_file=str(coef_file), scaler_file=str(scaler_file) if scaler_file else None)
            else:
                # Default to LinearClock for unknown clocks
                return LinearClock(coef_file=str(coef_file), scaler_file=str(scaler_file) if scaler_file else None)
        
        else:
            raise ValueError(f"Unknown clock type: {clock_info['type']}")
    
    def _find_weight_file(self, clock_info: Dict, variant: str, subtype: str) -> Path:
        """Find the appropriate weight file for the given parameters."""
        variants = clock_info['variants']
        
        # Handle single variant case (like kuo_2024)
        if 'default' in variants and len(variants) == 1:
            variant_info = variants['default']
            weight_files = list(variant_info['path'].glob("*.txt"))
            if not weight_files:
                raise FileNotFoundError(f"No weight files found in {variant_info['path']}")
            return weight_files[0]  # Return the first (and likely only) weight file
        
        # Handle multi-variant case (like goeminne_2025)
        if variant is None:
            variant = list(variants.keys())[0]  # Use first variant as default
        
        if variant not in variants:
            available_variants = list(variants.keys())
            raise ValueError(f"Variant '{variant}' not found. Available: {available_variants}")
        
        variant_info = variants[variant]
        variant_path = variant_info['path']
        
        # Look for the appropriate weight file
        if subtype == "conventional":
            # Look for files with "conventional" in the name
            conventional_files = list(variant_path.glob("*conventional*.txt"))
            if conventional_files:
                return conventional_files[0]
        else:
            # Look for organ-specific files
            organ_files = list(variant_path.glob(f"*{subtype}*.txt"))
            if organ_files:
                return organ_files[0]
        
        # Fallback: return any .txt file
        all_files = list(variant_path.glob("*.txt"))
        if all_files:
            return all_files[0]
        
        raise FileNotFoundError(f"No suitable weight file found for variant '{variant}', subtype '{subtype}'")
    
    def _find_scaler_file(self, scaler_name: str) -> Optional[Path]:
        """Find the scaler file by name."""
        scalers = self._discover_scalers()
        
        if scaler_name in scalers:
            return scalers[scaler_name]
        
        available_scalers = list(scalers.keys())
        raise ValueError(f"Scaler '{scaler_name}' not found. Available: {available_scalers}")
    
    def _parse_clock_name(self, name: str) -> tuple[str, Optional[str]]:
        """
        Parse a clock name to extract base name and variant.
        
        Examples:
            'kuo_2024' -> ('kuo_2024', None)
            'goeminne_2025' -> ('goeminne_2025', None)  # defaults to first variant
            'goeminne_2025_full_chrono' -> ('goeminne_2025', 'full_chrono')
            'goeminne_2025_reduced_mortality' -> ('goeminne_2025', 'reduced_mortality')
        """
        clocks = self._discover_clocks()
        
        # First check if the full name matches a base clock
        if name in clocks:
            return name, None
        
        # Try to parse extended names
        parts = name.split('_')
        if len(parts) >= 3:
            # Try different combinations to find base name
            for i in range(2, len(parts)):
                potential_base = '_'.join(parts[:i])
                potential_variant = '_'.join(parts[i:])
                
                if potential_base in clocks:
                    # Check if the variant actually exists for this clock
                    clock_info = clocks[potential_base]
                    if clock_info['type'] == 'simple' and potential_variant in clock_info['variants']:
                        return potential_base, potential_variant
        
        # If no parsing worked, assume it's a base name (will cause error later if not found)
        return name, None

    def _get_all_available_clock_names(self) -> List[str]:
        """Get all available clock names including base names and extended variant names."""
        clocks = self._discover_clocks()
        all_names = []
        
        for base_name, clock_info in clocks.items():
            all_names.append(base_name)  # Add base name
            
            # Add extended names for simple clocks with variants
            if clock_info['type'] == 'simple' and len(clock_info['variants']) > 1:
                for variant_name in clock_info['variants']:
                    if variant_name != 'default':
                        extended_name = f"{base_name}_{variant_name}"
                        all_names.append(extended_name)
        
        return sorted(all_names)
    
    def list_available_clocks(self) -> List[str]:
        """Return a list of all available clock names including extended variant names."""
        return self._get_all_available_clock_names()

    def list_base_clocks(self) -> List[str]:
        """Return a list of base clock names only."""
        clocks = self._discover_clocks()
        return list(clocks.keys())