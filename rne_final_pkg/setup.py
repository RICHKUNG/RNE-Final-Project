from setuptools import setup, find_packages
import os
from glob import glob

package_name = 'rne_final_pkg'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools', 'pyyaml'],
    zip_safe=True,
    entry_points={
        'console_scripts': [
            'final_mission = rne_final_pkg.final_mission:main',
            'scripted_final_mission = rne_final_pkg.scripted_final_mission:main',
            'door_test     = rne_final_pkg.door_test:main',
            'ramp_bear     = rne_final_pkg.ramp_bear:main',
            'topic_check   = rne_final_pkg.topic_check:main',
            'yolo_align    = rne_final_pkg.yolo_align:main',
            'get_bear      = rne_final_pkg.get_bear_node:main',
            'build_map     = rne_final_pkg.final_mapping_manager:main',
            'rect_map      = rne_final_pkg.rectangle_mapping:main',
        ],
    },
)
