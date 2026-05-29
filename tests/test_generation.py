from pathlib import Path
import json
import unittest

from PIL import Image

from rocky.config import BatchConfig, ParameterRanges
from rocky.pipeline import RockGenerator


class GenerationTests(unittest.TestCase):
    def test_generates_mesh_texture_and_reports(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as raw_tmp:
            tmp_path = Path(raw_tmp)
            texture_dir = tmp_path / "textures" / "test_rock" / "textures"
            texture_dir.mkdir(parents=True)
            Image.new("RGB", (4, 4), (90, 88, 80)).save(texture_dir / "test_rock_diff_4k.jpg")
            Image.new("RGB", (4, 4), (128, 128, 255)).save(texture_dir / "test_rock_nor_gl_4k.png")
            Image.new("RGB", (4, 4), (80, 80, 80)).save(texture_dir / "test_rock_rough_4k.jpg")
            Image.new("L", (4, 4), 128).save(texture_dir / "test_rock_disp_4k.png")
            config = BatchConfig(
                seed=42,
                count=2,
                output_dir=tmp_path,
                texture_dir=tmp_path / "textures",
                export_formats=("obj", "glb"),
                ranges=ParameterRanges(fracture_count=(2, 3), crack_count=(2, 4)),
            )

            states = RockGenerator(config).generate_batch()

            self.assertEqual(len(states), 2)
            self.assertIsNotNone(states[0].mesh)
            assert states[0].mesh is not None
            self.assertGreater(states[0].mesh.vertex_count(), 0)
            self.assertGreater(states[0].mesh.face_count(), 0)
            self.assertIn("diffuse", states[0].material_maps)
            bounds_min, bounds_max = states[0].mesh.bounds()
            self.assertLessEqual(bounds_max.y - bounds_min.y, config.max_height + 1e-6)
            self.assertTrue(states[0].params.size_class)
            self.assertTrue(states[0].params.archetype)
            self.assertTrue((tmp_path / "rock_000" / "rock_000.obj").exists())
            self.assertTrue((tmp_path / "rock_000" / "rock_000.glb").exists())
            self.assertTrue((tmp_path / "rock_000" / "material.json").exists())
            self.assertFalse((tmp_path / "rock_000" / "rock_000_diffuse.jpg").exists())
            self.assertTrue((tmp_path / "preview.png").exists())
            self.assertTrue((tmp_path / "report.json").exists())
            self.assertTrue((tmp_path / "rock_000" / "info.json").exists())
            obj_text = (tmp_path / "rock_000" / "rock_000.obj").read_text(encoding="utf-8")
            self.assertIn("s off\n", obj_text)
            self.assertIn("/1", obj_text)
            mtl_text = (tmp_path / "rock_000" / "rock_000.mtl").read_text(encoding="utf-8")
            self.assertIn("../textures/test_rock/textures/test_rock_diff_4k.jpg", mtl_text)
            self.assertIn("map_Kd", mtl_text)
            self.assertIn("map_Bump", mtl_text)
            self.assertIn("map_Pr", mtl_text)
            self.assertIn("disp", mtl_text)
            info = json.loads((tmp_path / "rock_000" / "info.json").read_text(encoding="utf-8"))
            self.assertEqual(info["placement"]["size_class"], states[0].params.size_class)
            self.assertEqual(info["placement"]["archetype"], states[0].params.archetype)
            self.assertEqual(info["placement"]["shape_type"], states[0].params.shape_type)
            self.assertEqual(info["placement"]["material_type"], states[0].params.material_type)
            self.assertEqual(info["placement"]["placement_role"], states[0].params.placement_role)
            self.assertIn("footprint_radius_m", info["placement"])
            self.assertIn("drive_over", info["placement"])
            self.assertIn("collision_category", info["placement"])
            self.assertIn("orientation", info["placement"])
            self.assertEqual(info["placement"]["orientation"]["local_up_axis"], "Y")
            self.assertTrue(info["placement"]["orientation"]["allow_random_yaw"])
            self.assertGreaterEqual(info["placement"]["orientation"]["max_pitch_deg"], 0.0)
            self.assertGreaterEqual(info["placement"]["orientation"]["max_roll_deg"], 0.0)
            self.assertIn("obj", info["exports"])
            self.assertIn("glb", info["exports"])
            self.assertIn("material", info["exports"])
            self.assertEqual(len(states[0].mesh.face_uvs), states[0].mesh.face_count())
            material = json.loads((tmp_path / "rock_000" / "material.json").read_text(encoding="utf-8"))
            self.assertEqual(material["schema"], "rocky.material.v1")
            self.assertEqual(material["maps"]["diffuse"]["usage"], "base_color")
            self.assertEqual(material["maps"]["diffuse"]["color_space"], "sRGB")
            self.assertEqual(material["maps"]["normal"]["normal_convention"], "OpenGL")
            self.assertEqual(material["maps"]["roughness"]["color_space"], "linear")
            self.assertEqual(material["maps"]["displacement"]["usage"], "displacement")

    def test_config_loads_example(self) -> None:
        config = BatchConfig.from_file("configs/config.json")

        self.assertEqual(config.count, 36)
        self.assertEqual(config.max_height, 2.0)
        self.assertGreater(config.resolution_scale, 0.0)
        self.assertEqual(config.export_formats, ("obj",))
        self.assertGreaterEqual(len(config.size_classes), 4)
        self.assertGreaterEqual(len(config.archetypes), 6)
        subdivisions = {profile["name"]: profile["subdivisions"] for profile in config.size_classes}
        self.assertLess(subdivisions["floor_pebble"], subdivisions["floor_cobble"])
        self.assertLess(subdivisions["floor_cobble"], subdivisions["step_rock"])
        self.assertLess(subdivisions["step_rock"], subdivisions["rover_obstacle"])

    def test_resolution_scale_changes_sampled_subdivisions(self) -> None:
        low = BatchConfig.from_mapping(
            {
                "count": 1,
                "resolution_scale": 0.5,
                "size_classes": [{"name": "fixed", "weight": 1, "height": [1, 1], "diameter": [1, 1], "subdivisions": 4}],
                "archetypes": [{"name": "fixed", "weight": 1}],
            }
        )
        high = BatchConfig.from_mapping(
            {
                "count": 1,
                "resolution_scale": 1.5,
                "size_classes": [{"name": "fixed", "weight": 1, "height": [1, 1], "diameter": [1, 1], "subdivisions": 4}],
                "archetypes": [{"name": "fixed", "weight": 1}],
            }
        )

        low_state = RockGenerator(low)._sample_parameters(0, __import__("random").Random(1))
        high_state = RockGenerator(high)._sample_parameters(0, __import__("random").Random(1))

        self.assertEqual(low_state.subdivisions, 2)
        self.assertEqual(high_state.subdivisions, 6)
