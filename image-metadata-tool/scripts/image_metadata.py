#!/usr/bin/env python3
"""
Image Metadata Tool - Extract, analyze, and manage EXIF metadata from images.
"""

import argparse
import json
import os
from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS

try:
    import folium
    from folium.plugins import MarkerCluster
    FOLIUM_AVAILABLE = True
except ImportError:
    FOLIUM_AVAILABLE = False


class ImageMetadata:
    """Extract and manage image EXIF metadata."""

    SUPPORTED_FORMATS = {'.jpg', '.jpeg', '.tiff', '.tif', '.png', '.webp', '.heic', '.heif'}

    def __init__(self):
        """Initialize the metadata extractor."""
        self.image: Optional[Image.Image] = None
        self.filepath: Optional[str] = None
        self.exif_data: Dict = {}

    def load(self, filepath: str) -> 'ImageMetadata':
        """
        Load an image file.

        Args:
            filepath: Path to image file

        Returns:
            Self for method chaining
        """
        self.filepath = filepath
        self.image = Image.open(filepath)
        self._extract_exif()
        return self

    def _extract_exif(self):
        """Extract EXIF data from loaded image."""
        self.exif_data = {}

        if self.image is None:
            return

        exif = self.image.getexif()
        if exif is None:
            return

        # Standard EXIF tags
        for tag_id, value in exif.items():
            tag = TAGS.get(tag_id, tag_id)
            self.exif_data[tag] = value

        # IFD data (more detailed info)
        for ifd_id in [0x8769, 0x8825]:  # EXIF IFD, GPS IFD
            try:
                ifd = exif.get_ifd(ifd_id)
                if ifd:
                    tag_mapping = GPSTAGS if ifd_id == 0x8825 else TAGS
                    for tag_id, value in ifd.items():
                        tag = tag_mapping.get(tag_id, tag_id)
                        self.exif_data[tag] = value
            except Exception:
                pass

    def extract(self) -> Dict:
        """
        Extract all relevant metadata.

        Returns:
            Dictionary with organized metadata
        """
        if self.image is None:
            raise ValueError("No image loaded")

        return {
            "file": self._get_file_info(),
            "camera": self.get_camera_info(),
            "settings": self._get_capture_settings(),
            "datetime": self.get_datetime(),
            "gps": self.get_gps(),
            "dimensions": self.get_dimensions()
        }

    def _get_file_info(self) -> Dict:
        """Get file information."""
        if self.filepath is None:
            return {}

        path = Path(self.filepath)
        stat = path.stat()

        return {
            "name": path.name,
            "path": str(path.absolute()),
            "size": stat.st_size,
            "format": self.image.format if self.image else None
        }

    def get_camera_info(self) -> Dict:
        """
        Get camera and lens information.

        Returns:
            Dictionary with camera details
        """
        info = {}

        mappings = {
            "make": ["Make"],
            "model": ["Model"],
            "lens": ["LensModel", "LensInfo"],
            "lens_id": ["LensSerialNumber"],
            "software": ["Software"],
            "serial_number": ["BodySerialNumber"]
        }

        for key, tags in mappings.items():
            for tag in tags:
                if tag in self.exif_data:
                    value = self.exif_data[tag]
                    if isinstance(value, bytes):
                        try:
                            value = value.decode('utf-8').strip('\x00')
                        except Exception:
                            continue
                    info[key] = str(value).strip()
                    break

        return info

    def _get_capture_settings(self) -> Dict:
        """Get capture settings (exposure, ISO, etc.)."""
        settings = {}

        # Exposure time
        if "ExposureTime" in self.exif_data:
            exp = self.exif_data["ExposureTime"]
            if isinstance(exp, tuple):
                settings["exposure_time"] = f"{exp[0]}/{exp[1]}" if exp[1] != 1 else str(exp[0])
            else:
                settings["exposure_time"] = str(exp)

        # F-number
        if "FNumber" in self.exif_data:
            fn = self.exif_data["FNumber"]
            if isinstance(fn, tuple):
                settings["f_number"] = fn[0] / fn[1] if fn[1] != 0 else fn[0]
            else:
                settings["f_number"] = float(fn)

        # ISO
        if "ISOSpeedRatings" in self.exif_data:
            iso = self.exif_data["ISOSpeedRatings"]
            settings["iso"] = iso[0] if isinstance(iso, tuple) else iso

        # Focal length
        if "FocalLength" in self.exif_data:
            fl = self.exif_data["FocalLength"]
            if isinstance(fl, tuple):
                settings["focal_length"] = fl[0] / fl[1] if fl[1] != 0 else fl[0]
            else:
                settings["focal_length"] = float(fl)

        if "FocalLengthIn35mmFilm" in self.exif_data:
            settings["focal_length_35mm"] = self.exif_data["FocalLengthIn35mmFilm"]

        # Exposure program
        exposure_programs = {
            0: "Not defined", 1: "Manual", 2: "Program", 3: "Aperture priority",
            4: "Shutter priority", 5: "Creative", 6: "Action", 7: "Portrait", 8: "Landscape"
        }
        if "ExposureProgram" in self.exif_data:
            settings["exposure_program"] = exposure_programs.get(
                self.exif_data["ExposureProgram"], "Unknown"
            )

        # Metering mode
        metering_modes = {
            0: "Unknown", 1: "Average", 2: "Center-weighted", 3: "Spot",
            4: "Multi-spot", 5: "Pattern", 6: "Partial"
        }
        if "MeteringMode" in self.exif_data:
            settings["metering_mode"] = metering_modes.get(
                self.exif_data["MeteringMode"], "Unknown"
            )

        # Flash
        if "Flash" in self.exif_data:
            flash = self.exif_data["Flash"]
            settings["flash"] = "Flash fired" if (flash & 1) else "No flash"

        # White balance
        if "WhiteBalance" in self.exif_data:
            wb = self.exif_data["WhiteBalance"]
            settings["white_balance"] = "Manual" if wb == 1 else "Auto"

        return settings

    def get_datetime(self) -> Dict:
        """
        Get timestamp information.

        Returns:
            Dictionary with datetime details
        """
        dt_info = {}

        datetime_tags = {
            "original": "DateTimeOriginal",
            "digitized": "DateTimeDigitized",
            "modified": "DateTime"
        }

        for key, tag in datetime_tags.items():
            if tag in self.exif_data:
                value = self.exif_data[tag]
                if isinstance(value, bytes):
                    value = value.decode('utf-8')
                dt_info[key] = str(value).strip('\x00')

        # Timezone
        if "OffsetTimeOriginal" in self.exif_data:
            dt_info["timezone"] = self.exif_data["OffsetTimeOriginal"]

        return dt_info

    def get_gps(self) -> Optional[Dict]:
        """
        Get GPS coordinates and related data.

        Returns:
            Dictionary with GPS data or None if not available
        """
        if "GPSLatitude" not in self.exif_data or "GPSLongitude" not in self.exif_data:
            return None

        def convert_to_degrees(value):
            """Convert GPS coordinates to decimal degrees."""
            if isinstance(value, tuple) and len(value) == 3:
                d = float(value[0]) if not isinstance(value[0], tuple) else value[0][0] / value[0][1]
                m = float(value[1]) if not isinstance(value[1], tuple) else value[1][0] / value[1][1]
                s = float(value[2]) if not isinstance(value[2], tuple) else value[2][0] / value[2][1]
                return d + m / 60 + s / 3600
            return float(value)

        try:
            lat = convert_to_degrees(self.exif_data["GPSLatitude"])
            lon = convert_to_degrees(self.exif_data["GPSLongitude"])

            # Apply reference (N/S, E/W)
            if self.exif_data.get("GPSLatitudeRef", "N") == "S":
                lat = -lat
            if self.exif_data.get("GPSLongitudeRef", "E") == "W":
                lon = -lon

            gps_info = {
                "latitude": lat,
                "longitude": lon,
                "maps_url": f"https://maps.google.com/maps?q={lat},{lon}"
            }

            # Altitude
            if "GPSAltitude" in self.exif_data:
                alt = self.exif_data["GPSAltitude"]
                if isinstance(alt, tuple):
                    alt = alt[0] / alt[1] if alt[1] != 0 else alt[0]
                gps_info["altitude"] = float(alt)

                alt_ref = self.exif_data.get("GPSAltitudeRef", 0)
                gps_info["altitude_ref"] = "Below sea level" if alt_ref == 1 else "Above sea level"

            # Direction
            if "GPSImgDirection" in self.exif_data:
                direction = self.exif_data["GPSImgDirection"]
                if isinstance(direction, tuple):
                    direction = direction[0] / direction[1] if direction[1] != 0 else direction[0]
                gps_info["direction"] = float(direction)

            return gps_info

        except Exception:
            return None

    def get_dimensions(self) -> Dict:
        """
        Get image dimensions and orientation.

        Returns:
            Dictionary with dimension info
        """
        if self.image is None:
            return {}

        dims = {
            "width": self.image.width,
            "height": self.image.height,
            "megapixels": round(self.image.width * self.image.height / 1_000_000, 1)
        }

        # Orientation
        orientation_map = {
            1: "Horizontal",
            2: "Horizontal (flipped)",
            3: "Rotated 180",
            4: "Rotated 180 (flipped)",
            5: "Rotated 90 CCW (flipped)",
            6: "Rotated 90 CW",
            7: "Rotated 90 CW (flipped)",
            8: "Rotated 90 CCW"
        }

        if "Orientation" in self.exif_data:
            dims["orientation"] = orientation_map.get(
                self.exif_data["Orientation"], "Unknown"
            )
        else:
            dims["orientation"] = "Horizontal"

        # Resolution
        if "XResolution" in self.exif_data:
            res = self.exif_data["XResolution"]
            dims["resolution_x"] = res[0] / res[1] if isinstance(res, tuple) else res

        if "YResolution" in self.exif_data:
            res = self.exif_data["YResolution"]
            dims["resolution_y"] = res[0] / res[1] if isinstance(res, tuple) else res

        res_units = {1: "none", 2: "inch", 3: "cm"}
        if "ResolutionUnit" in self.exif_data:
            dims["resolution_unit"] = res_units.get(
                self.exif_data["ResolutionUnit"], "unknown"
            )

        return dims

    def get_all_exif(self) -> Dict:
        """
        Get all raw EXIF data.

        Returns:
            Dictionary with all EXIF tags
        """
        result = {}
        for key, value in self.exif_data.items():
            if isinstance(value, bytes):
                try:
                    value = value.decode('utf-8').strip('\x00')
                except Exception:
                    value = f"<binary: {len(value)} bytes>"
            result[str(key)] = value
        return result

    def has_location(self) -> bool:
        """
        Check if image has GPS data.

        Returns:
            True if GPS coordinates are present
        """
        return self.get_gps() is not None

    def strip_metadata(self, output: str, keep_orientation: bool = True) -> str:
        """
        Remove EXIF metadata from image.

        Args:
            output: Output file path
            keep_orientation: Whether to preserve orientation tag

        Returns:
            Output file path
        """
        if self.image is None:
            raise ValueError("No image loaded")

        # Create new image without EXIF
        data = list(self.image.getdata())
        img_no_exif = Image.new(self.image.mode, self.image.size)
        img_no_exif.putdata(data)

        # Apply rotation if needed and keeping orientation
        if keep_orientation and "Orientation" in self.exif_data:
            orientation = self.exif_data["Orientation"]
            rotation_map = {
                3: 180,
                6: -90,
                8: 90
            }
            if orientation in rotation_map:
                img_no_exif = img_no_exif.rotate(
                    rotation_map[orientation], expand=True
                )

        # Save without EXIF
        img_no_exif.save(output, quality=95)
        return output

    def extract_batch(self, folder: str, recursive: bool = False) -> List[Dict]:
        """
        Extract metadata from all images in folder.

        Args:
            folder: Folder path
            recursive: Include subfolders

        Returns:
            List of metadata dictionaries
        """
        results = []
        path = Path(folder)

        if recursive:
            files = path.rglob("*")
        else:
            files = path.glob("*")

        for file in files:
            if file.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    self.load(str(file))
                    results.append(self.extract())
                except Exception as e:
                    results.append({
                        "file": {"name": file.name, "path": str(file), "error": str(e)}
                    })

        return results

    def strip_batch(self, input_folder: str, output_folder: str) -> List[Dict]:
        """
        Strip metadata from all images in folder.

        Args:
            input_folder: Input folder path
            output_folder: Output folder path

        Returns:
            List of processed files
        """
        results = []
        input_path = Path(input_folder)
        output_path = Path(output_folder)
        output_path.mkdir(parents=True, exist_ok=True)

        for file in input_path.glob("*"):
            if file.suffix.lower() in self.SUPPORTED_FORMATS:
                try:
                    self.load(str(file))
                    out_file = output_path / file.name
                    self.strip_metadata(str(out_file))
                    results.append({
                        "input": str(file),
                        "output": str(out_file),
                        "success": True
                    })
                except Exception as e:
                    results.append({
                        "input": str(file),
                        "error": str(e),
                        "success": False
                    })

        return results

    def generate_map(self, images: List[Dict], output: str) -> str:
        """
        Generate an interactive map from geotagged images.

        Args:
            images: List of image metadata dictionaries
            output: Output HTML file path

        Returns:
            Output file path
        """
        if not FOLIUM_AVAILABLE:
            raise ImportError("folium is required for map generation")

        # Filter images with GPS
        geotagged = [img for img in images if img.get("gps")]

        if not geotagged:
            raise ValueError("No geotagged images found")

        # Calculate center
        lats = [img["gps"]["latitude"] for img in geotagged]
        lons = [img["gps"]["longitude"] for img in geotagged]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)

        # Create map
        m = folium.Map(location=[center_lat, center_lon], zoom_start=10)

        # Add markers
        marker_cluster = MarkerCluster()

        for img in geotagged:
            gps = img["gps"]
            file_info = img.get("file", {})
            camera = img.get("camera", {})
            dt = img.get("datetime", {})

            popup_html = f"""
            <b>{file_info.get('name', 'Unknown')}</b><br>
            Camera: {camera.get('model', 'Unknown')}<br>
            Date: {dt.get('original', 'Unknown')}<br>
            <a href="{gps.get('maps_url', '#')}" target="_blank">Open in Google Maps</a>
            """

            folium.Marker(
                location=[gps["latitude"], gps["longitude"]],
                popup=folium.Popup(popup_html, max_width=300),
                icon=folium.Icon(color='blue', icon='camera', prefix='fa')
            ).add_to(marker_cluster)

        marker_cluster.add_to(m)

        # Save map
        m.save(output)
        return output

    def to_json(self, output: str) -> str:
        """Export metadata to JSON file."""
        data = self.extract()
        with open(output, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return output

    def to_csv(self, output: str) -> str:
        """Export metadata to CSV (requires previous batch extraction)."""
        raise NotImplementedError("Use extract_batch() and pandas for CSV export")


def main():
    parser = argparse.ArgumentParser(
        description="Image Metadata Tool - Extract and manage EXIF metadata"
    )

    parser.add_argument("--input", "-i", required=True, help="Input image or folder")
    parser.add_argument("--output", "-o", help="Output file or folder")
    parser.add_argument("--gps", action="store_true", help="Show GPS information")
    parser.add_argument("--strip", action="store_true", help="Strip metadata from image")
    parser.add_argument("--map", help="Generate location map (output HTML path)")
    parser.add_argument("--recursive", "-r", action="store_true", help="Process subfolders")
    parser.add_argument("--fields", help="Specific fields to show (comma-separated)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--all", "-a", action="store_true", help="Show all EXIF data")

    args = parser.parse_args()

    meta = ImageMetadata()
    input_path = Path(args.input)

    if input_path.is_dir():
        # Batch processing
        results = meta.extract_batch(args.input, recursive=args.recursive)

        if args.map:
            meta.generate_map(results, args.map)
            print(f"Map generated: {args.map}")

        elif args.strip and args.output:
            strip_results = meta.strip_batch(args.input, args.output)
            success = sum(1 for r in strip_results if r["success"])
            print(f"Stripped metadata from {success}/{len(strip_results)} images")

        elif args.json:
            print(json.dumps(results, indent=2, default=str))

        else:
            # Summary
            geotagged = sum(1 for r in results if r.get("gps"))
            print(f"Processed {len(results)} images")
            print(f"Geotagged: {geotagged}")

            if args.output:
                import pandas as pd
                flat_results = []
                for r in results:
                    flat = {
                        "filename": r.get("file", {}).get("name"),
                        "camera_make": r.get("camera", {}).get("make"),
                        "camera_model": r.get("camera", {}).get("model"),
                        "datetime": r.get("datetime", {}).get("original"),
                        "latitude": r.get("gps", {}).get("latitude") if r.get("gps") else None,
                        "longitude": r.get("gps", {}).get("longitude") if r.get("gps") else None,
                        "width": r.get("dimensions", {}).get("width"),
                        "height": r.get("dimensions", {}).get("height")
                    }
                    flat_results.append(flat)
                pd.DataFrame(flat_results).to_csv(args.output, index=False)
                print(f"Saved to: {args.output}")

    else:
        # Single file
        meta.load(args.input)

        if args.strip:
            output = args.output or f"clean_{input_path.name}"
            meta.strip_metadata(output)
            print(f"Metadata stripped: {output}")

        elif args.all:
            data = meta.get_all_exif()
            if args.json:
                print(json.dumps(data, indent=2, default=str))
            else:
                for key, value in data.items():
                    print(f"{key}: {value}")

        elif args.gps:
            gps = meta.get_gps()
            if gps:
                if args.json:
                    print(json.dumps(gps, indent=2))
                else:
                    print(f"Latitude: {gps['latitude']}")
                    print(f"Longitude: {gps['longitude']}")
                    if 'altitude' in gps:
                        print(f"Altitude: {gps['altitude']} m")
                    print(f"Maps: {gps['maps_url']}")
            else:
                print("No GPS data found")

        else:
            data = meta.extract()

            if args.fields:
                fields = args.fields.split(',')
                filtered = {}
                for field in fields:
                    if field in data:
                        filtered[field] = data[field]
                data = filtered

            if args.json:
                print(json.dumps(data, indent=2, default=str))
            else:
                print(f"\n=== {data['file']['name']} ===\n")

                if data.get("camera"):
                    print("CAMERA")
                    for k, v in data["camera"].items():
                        print(f"  {k}: {v}")

                if data.get("settings"):
                    print("\nSETTINGS")
                    for k, v in data["settings"].items():
                        print(f"  {k}: {v}")

                if data.get("datetime"):
                    print("\nDATETIME")
                    for k, v in data["datetime"].items():
                        print(f"  {k}: {v}")

                if data.get("gps"):
                    print("\nGPS")
                    for k, v in data["gps"].items():
                        print(f"  {k}: {v}")

                if data.get("dimensions"):
                    print("\nDIMENSIONS")
                    for k, v in data["dimensions"].items():
                        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
