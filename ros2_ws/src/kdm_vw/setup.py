from glob import glob
from pathlib import Path
from setuptools import find_packages, setup


package_name = "kdm_vw"


def package_files(pattern):
    return [path for path in glob(pattern) if Path(path).is_file()]


setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", package_files("launch/*.py") + package_files("launch/*.rviz")),
        (f"share/{package_name}/navigation_config", package_files("navigation_config/*")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Aadham Ahmad",
    maintainer_email="aa2212@hw.ac.uk",
    description="Standalone VGR dataset wrapper for KDM+V/W gas distribution mapping.",
    license="GPLv3",
    entry_points={
        "console_scripts": [
            "kdm_vw_mapper = kdm_vw.kdm_vw_mapper_node:main",
            "coverage_explorer = kdm_vw.coverage_explorer_node:main",
            "csv_to_heatmap = kdm_vw.csv_to_heatmap:main",
            "export_ground_truth_map = kdm_vw.export_ground_truth_map:main",
        ],
    },
)
