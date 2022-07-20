import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

PROJECT_NAME = "AIOPS_DL_usecase"
USER_NAME = "anilans029"

setuptools.setup(
    name= "src",
    version="0.0.1",
    author=USER_NAME,
    author_email="anilsai029@gmail.com",
    description="A small package for dvc_dl_pipeline_demo",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{USER_NAME}/{PROJECT_NAME}",
    packages=["src"],
    python_requires=">=3.7",
    install_requires=[
        'dvc',
        'pandas',
        'scikit-learn',
        'numpy',
        'tensorflow',
        'boto3',
        'PyYAML',
        'matplotlib',
        'tqdm'

    ]
)