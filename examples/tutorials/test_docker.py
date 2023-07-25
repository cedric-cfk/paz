import argparse

parser = argparse.ArgumentParser(description='Docker Entrypoint test')
parser.add_argument('-p', '--data_path', type=str,
                    default='BLA',
                    help='Default root data path')

args = parser.parse_args()

print("Hello World!")
print("Data path: {}".format(args.data_path))