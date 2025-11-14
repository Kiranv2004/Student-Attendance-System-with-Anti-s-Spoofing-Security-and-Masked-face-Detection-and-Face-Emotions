"""Check if all required packages are installed"""
import sys
import subprocess
import pkg_resources

# Read requirements.txt
requirements_file = "requirements.txt"
required_packages = {}

with open(requirements_file, 'r') as f:
    for line in f:
        line = line.strip()
        if line and not line.startswith('#'):
            if '==' in line:
                name, version = line.split('==')
                required_packages[name.lower()] = version

print("="*60)
print("CHECKING INSTALLED PACKAGES")
print("="*60)
print()

installed_packages = {pkg.key: pkg.version for pkg in pkg_resources.working_set}

missing_packages = []
version_mismatches = []
installed_correctly = []

for package_name, required_version in required_packages.items():
    if package_name not in installed_packages:
        missing_packages.append((package_name, required_version))
        print(f"[MISSING] {package_name}=={required_version}")
    elif installed_packages[package_name] != required_version:
        version_mismatches.append((package_name, required_version, installed_packages[package_name]))
        print(f"[VERSION MISMATCH] {package_name}")
        print(f"   Required: {required_version}")
        print(f"   Installed: {installed_packages[package_name]}")
    else:
        installed_correctly.append(package_name)
        print(f"[OK] {package_name}=={required_version}")

print()
print("="*60)
print("SUMMARY")
print("="*60)
print(f"[OK] Correctly installed: {len(installed_correctly)}")
print(f"[WARNING] Version mismatches: {len(version_mismatches)}")
print(f"[ERROR] Missing packages: {len(missing_packages)}")
print()

if missing_packages:
    print("MISSING PACKAGES:")
    for name, version in missing_packages:
        print(f"  - {name}=={version}")
    print()
    print("Install with:")
    print(f"  pip install {' '.join([f'{name}=={version}' for name, version in missing_packages])}")

if version_mismatches:
    print("VERSION MISMATCHES (may still work, but versions differ):")
    for name, required, installed in version_mismatches:
        print(f"  - {name}: required {required}, installed {installed}")

if not missing_packages and not version_mismatches:
    print("[SUCCESS] All packages are installed correctly!")
elif not missing_packages:
    print("\n[WARNING] Some packages have version mismatches, but they may still work.")
    print("   If you encounter issues, try installing exact versions:")
    for name, required, installed in version_mismatches:
        print(f"   pip install {name}=={required}")

