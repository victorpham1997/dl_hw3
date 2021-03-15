import argparse
from utils_ic import load_data, read_jason
# from model_ic import NN_Classifier, validation, make_NN, save_checkpoint
import model_ic
import model_ic_densenet169
import torch



parser = argparse.ArgumentParser(description="Train image classifier model")
parser.add_argument("data_dir", help="load data directory")
parser.add_argument("--category_names", default="cat_to_name.json", help="choose category names")
parser.add_argument("--arch", default="densenet169", help="choose model architecture")
parser.add_argument("--learning_rate", type=int, default=0.001, help="set learning rate")
parser.add_argument("--hidden_units", type=int, default=1024, help="set hidden units")
parser.add_argument("--epochs", type=int, default=1, help="set epochs")
parser.add_argument("--gpu", action="store_const", const="cuda", default="cpu", help="use gpu")
parser.add_argument("--finetuning_mode", type=int, default=0, help="set the training method: 0 for train from scratch, 1 finetuning classifier, 2 for finetuning everything")
parser.add_argument("--save_dir", help="save model")

args = parser.parse_args()

cat_to_name = read_jason(args.category_names)

trainloader, testloader, validloader, train_data = load_data(args.data_dir)

if args.arch == "densenet169":
  model = model_ic_densenet169.make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data, finetuning_mode = args.finetuning_mode)
  if args.save_dir:
    torch.save(model, args.save_dir)
#     model_ic_densenet169.save_checkpoint(model, args.save_dir)
    
    
elif args.arch == "resnet18":
  model = model_ic.make_NN(n_hidden=[args.hidden_units], n_epoch=args.epochs, labelsdict=cat_to_name, lr=args.learning_rate, device=args.gpu, \
                model_name=args.arch, trainloader=trainloader, validloader=validloader, train_data=train_data)
  if args.save_dir:
    model_ic.save_checkpoint(model, args.save_dir)
