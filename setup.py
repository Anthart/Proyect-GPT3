from setuptools import setup

setup(
    name="proyectT",
    version="0.0.1",
    description="Proyecto GPT3",
    author="Anthart",
    author_email="ffarteaga98@gmail.com",
    url="https://github.com/Anthart/Proyect-GPT3.git",
    packages=['proyect_modules'],
    zip_safe=False,
    install_requires=["transformers", "openai", "pandas", "scikit-learn"]
)
