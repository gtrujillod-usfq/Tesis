## setup.py
## Script de inicializacion para verificar que el proyecto esta configurado correctamente

import sys
import os
from pathlib import Path

## Colores ANSI (sin estilos complejos, solo textos basicos)
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'

def check_python_version():
    ## Verificar que Python >= 3.8
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"{Colors.GREEN}[OK]{Colors.END} Python {version.major}.{version.minor}.{version.micro}")
        return True
    else:
        print(f"{Colors.RED}[ERROR]{Colors.END} Python {version.major}.{version.minor} (requerido >= 3.8)")
        return False

def check_directories():
    ## Verificar que existan los directorios principales
    root = Path.cwd()
    required_dirs = ['src', 'tests', 'data', 'outputs', 'results', 'checkpoints']
    
    all_exist = True
    for dir_name in required_dirs:
        dir_path = root / dir_name
        if dir_path.exists():
            print(f"{Colors.GREEN}[OK]{Colors.END} Directorio: {dir_name}/")
        else:
            print(f"{Colors.YELLOW}[CREAR]{Colors.END} Directorio: {dir_name}/ (se creara automaticamente)")
            dir_path.mkdir(parents=True, exist_ok=True)
    
    return True

def check_files():
    ## Verificar que existan los archivos principales
    root = Path.cwd()
    required_files = [
        'src/__init__.py',
        'src/config.py',
        'src/utils.py',
        'src/coherence_metrics.py',
        'tests/test_area1_coherence.py',
        'main.ipynb',
        'README.md',
    ]
    
    all_exist = True
    for file_name in required_files:
        file_path = root / file_name
        if file_path.exists():
            size_kb = file_path.stat().st_size / 1024
            print(f"{Colors.GREEN}[OK]{Colors.END} {file_name:40s} ({size_kb:6.1f} KB)")
        else:
            print(f"{Colors.RED}[FALTA]{Colors.END} {file_name:40s}")
            all_exist = False
    
    return all_exist

def check_imports():
    ## Intentar importar modulos principales
    sys.path.insert(0, 'src')
    
    modules = [
        ('config', 'MammoVLMConfig'),
        ('utils', 'setup_logging'),
        ('coherence_metrics', 'AlignmentMetricsCollector'),
    ]
    
    all_ok = True
    for module_name, class_name in modules:
        try:
            module = __import__(module_name)
            obj = getattr(module, class_name, None)
            if obj:
                print(f"{Colors.GREEN}[OK]{Colors.END} from {module_name} import {class_name}")
            else:
                print(f"{Colors.RED}[ERROR]{Colors.END} {class_name} no encontrado en {module_name}")
                all_ok = False
        except ImportError as e:
            print(f"{Colors.RED}[ERROR]{Colors.END} No se pudo importar {module_name}: {e}")
            all_ok = False
    
    return all_ok

def check_packages():
    ## Verificar paquetes principales
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('sklearn', 'Scikit-learn'),
        ('scipy', 'SciPy'),
    ]
    
    all_ok = True
    for package_name, display_name in packages:
        try:
            __import__(package_name)
            print(f"{Colors.GREEN}[OK]{Colors.END} {display_name}")
        except ImportError:
            print(f"{Colors.YELLOW}[FALTA]{Colors.END} {display_name} (opcional pero recomendado)")
            all_ok = False
    
    return all_ok

def main():
    print("\n" + "="*70)
    print("VERIFICACION DE SETUP - MammoVLM V2")
    print("="*70 + "\n")
    
    print("1. Verificar version de Python")
    print("-" * 70)
    python_ok = check_python_version()
    print()
    
    print("2. Verificar directorios")
    print("-" * 70)
    dirs_ok = check_directories()
    print()
    
    print("3. Verificar archivos principales")
    print("-" * 70)
    files_ok = check_files()
    print()
    
    print("4. Verificar paquetes Python")
    print("-" * 70)
    packages_ok = check_packages()
    print()
    
    print("5. Verificar imports de modulos propios")
    print("-" * 70)
    imports_ok = check_imports()
    print()
    
    print("="*70)
    if python_ok and dirs_ok and files_ok and imports_ok:
        print(f"{Colors.GREEN}SETUP VERIFICADO - TODO LISTO PARA USAR{Colors.END}")
        print("\nProximos pasos:")
        print("  1. jupyter notebook main.ipynb")
        print("  2. Ejecutar las celdas en orden")
        return 0
    else:
        print(f"{Colors.YELLOW}ADVERTENCIAS O ERRORES ENCONTRADOS{Colors.END}")
        print("\nPor favor revisa los mensajes anteriores e instala los paquetes faltantes:")
        print("  pip install numpy pandas matplotlib seaborn scikit-learn scipy --break-system-packages")
        return 1
    
    print("="*70 + "\n")

if __name__ == "__main__":
    exit_code = main()
    exit(exit_code)
