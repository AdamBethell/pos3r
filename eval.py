from pos3r.evaluate import get_args_parser, evaluate

if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    evaluate(args)