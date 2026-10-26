"""Regression checks for command-line/YAML precedence without GPU imports."""
import argparse
import tempfile
import unittest
from pathlib import Path

from utils.config import parse_args_with_yaml


class ConfigTests(unittest.TestCase):
    def parse(self, content, extra=()):
        parser = argparse.ArgumentParser()
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--run_name', default='run1')
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text(content)
            return parse_args_with_yaml(parser, ['--config', str(config), *extra])

    def test_explicit_default_overrides_yaml(self):
        self.assertEqual(self.parse('epochs: 20', ['--epochs', '300']).epochs, 300)

    def test_yaml_and_defaults(self):
        args = self.parse('epochs: 20')
        self.assertEqual(args.epochs, 20)
        self.assertEqual(args.run_name, 'run1')

    def test_equals_option(self):
        self.assertEqual(self.parse('epochs: 20', ['--epochs=300']).epochs, 300)

    def test_evaluation_alias_respects_explicit_cli(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--total_interp_steps', type=int, default=20)
        with tempfile.TemporaryDirectory() as directory:
            config = Path(directory) / 'config.yaml'
            config.write_text('total_interp_steps_train: 8\nmodel_training_only: true\n')
            options = dict(ignore_unknown=True, aliases={
                'total_interp_steps_train': 'total_interp_steps',
            })
            args = parse_args_with_yaml(parser, ['--config', str(config)], **options)
            self.assertEqual(args.total_interp_steps, 8)
            args = parse_args_with_yaml(parser, [
                '--config', str(config), '--total_interp_steps', '20',
            ], **options)
            self.assertEqual(args.total_interp_steps, 20)

    def test_unknown_key(self):
        with self.assertRaises(KeyError):
            self.parse('epohcs: 20')

    def test_non_mapping(self):
        for content in ['[]', 'false', '0']:
            with self.assertRaises(ValueError):
                self.parse(content)


if __name__ == '__main__':
    unittest.main()
