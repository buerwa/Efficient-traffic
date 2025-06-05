from setuptools import setup, find_packages

setup(
    name="traffic_monitor",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "fastapi",
        "uvicorn",
        "python-jose[cryptography]",
        "passlib[bcrypt]",
        "python-multipart",
        "sqlite3",
    ],
)
