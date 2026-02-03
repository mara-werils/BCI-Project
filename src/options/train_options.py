from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    """This class includes training options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        # visdom and HTML visualization parameters
        parser.add_argument('--display_freq', type=int, default=400, help='frequency of showing training results on screen')
        parser.add_argument('--display_ncols', type=int, default=4, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        parser.add_argument('--display_server', type=str, default="http://localhost", help='visdom server of the web display')
        parser.add_argument('--display_env', type=str, default='LLVIP2', help='visdom display environment name (default is "main")')
        parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        parser.add_argument('--update_html_freq', type=int, default=1000, help='frequency of saving training results to html')
        parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
        # network saving and loading parameters
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--save_by_iter', action='store_true', help='whether saves model by iteration')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
        parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        # training parameters
        parser.add_argument('--n_epochs', type=int, default=100, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        parser.add_argument('--lr_policy', type=str, default='linear', help='learning rate policy. [linear | step | plateau | cosine]')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')

        parser.add_argument('--which_perceptual', type=str, default='4_2', help='relu5_2 or relu4_2')
        parser.add_argument('--weight_perceptual', type=float, default=0.00003)
        parser.add_argument('--vgg_normal_correct', action='store_true', help='if true, correct vgg normalization and replace vgg FM model with ctx model')
        parser.add_argument('--use_22ctx', action='store_true', help='if true, also use 2-2 in ctx loss')
        parser.add_argument('--lambda_vgg', type=float, default=2, help='weight for vgg loss')
        parser.add_argument('--ctx_w', type=float, default=1.0, help='ctx loss weight')
        parser.add_argument('--PONO', action='store_true', help='use positional normalization ')
        parser.add_argument('--weight_conv', type=float, default=100)
        parser.add_argument('--weight_sobel', type=float, default=50)
        parser.add_argument('--weight_L2', type=float, default=25)
        parser.add_argument('--weight_L3', type=float, default=25)
        parser.add_argument('--weight_L4', type=float, default=25)

        # ============ Transfer Learning Options ============
        parser.add_argument('--pretrained_path', type=str, default='',
                            help='Path to pre-trained generator weights for transfer learning')
        parser.add_argument('--pretrained_D_path', type=str, default='',
                            help='Path to pre-trained discriminator weights')
        parser.add_argument('--freeze_encoder', action='store_true',
                            help='Freeze encoder layers during fine-tuning')
        parser.add_argument('--freeze_epochs', type=int, default=10,
                            help='Number of epochs to keep encoder frozen')
        parser.add_argument('--finetune_lr_factor', type=float, default=0.1,
                            help='LR multiplier for fine-tuning (e.g., 0.1 = 10x lower LR)')
        
        # ============ Data Augmentation Options ============
        parser.add_argument('--strong_augment', action='store_true',
                            help='Enable strong augmentation for small datasets')
        parser.add_argument('--mixup_alpha', type=float, default=0.0,
                            help='MixUp alpha parameter (0 = disabled)')
        
        # ============ HER2 Classification Options ============
        parser.add_argument('--enable_classification', action='store_true',
                            help='Enable HER2 classification head')
        parser.add_argument('--lambda_classifier', type=float, default=1.0,
                            help='Weight for classification loss')
        parser.add_argument('--classifier_only', action='store_true',
                            help='Train only classifier (freeze generator)')
        parser.add_argument('--num_classes', type=int, default=4,
                            help='Number of HER2 classes (default: 4 for 0, 1+, 2+, 3+)')
        parser.add_argument('--class_weighted_loss', action='store_true',
                            help='Use class weights for imbalanced classification')
        
        # ============ Logging Options ============
        parser.add_argument('--log_dir', type=str, default='./experiments/logs',
                            help='Directory for tensorboard logs')
        parser.add_argument('--use_tensorboard', action='store_true',
                            help='Use tensorboard for logging')

        self.isTrain = True
        return parser
