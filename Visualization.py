import os
import torch
import numpy as np
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import transforms
from dataset.MetaPlaces_Extra69 import get_Places_Extra69_dataloader, Places_Extra69_Online
# Cross-resolution Tasks
from models.teacherNet import Teacher
from models.studentNet import CNN_RIS

from distiller_zoo import ATD, Sample_entropy

batch_size = 256
train_loader, val_loader = get_Places_Extra69_dataloader(batch_size=batch_size, num_workers=0)
n_cls = 69
model_s = CNN_RIS(num_classes=n_cls).cuda()
model_t = Teacher(num_classes=n_cls).cuda()
model_t.load_state_dict(torch.load('save/models/Places_Extra69_Teacher/Best_Teacher_model.t7')['tnet'])

temperature = Sample_entropy('Places_Extra69', 18, batch_size, 0).cuda()(model_t.cuda())

# tensor1 = temperature[0].unsqueeze(0).unsqueeze(0)
# tensor2 = temperature[1000].unsqueeze(0).unsqueeze(0)
# tensor3 = temperature[51000].unsqueeze(0).unsqueeze(0)
# tensor4 = temperature[31000].unsqueeze(0).unsqueeze(0)
# tensor5 = temperature[61000].unsqueeze(0).unsqueeze(0)
# tensor6 = temperature[-1].unsqueeze(0).unsqueeze(0)
# temperature = torch.cat((tensor1, tensor2, tensor3, tensor4, tensor5, tensor6), dim=1)
# print(temperature)
# print(temperature.shape)
# plt.figure("temperature")
# plt.imshow(temperature.cpu())
# plt.axis('off')
# if not os.path.isdir('picture/'):
#     os.mkdir('picture/')
# plt.savefig('picture/temperature.jpg', dpi=500)
# plt.show()



transform_train = transforms.Compose([
        transforms.RandomCrop(92),
        transforms.RandomHorizontalFlip(),
])

transforms_teacher_train_Normalize = transforms.Normalize((0.44738832, 0.41790622, 0.37717262),
                                                          (0.2686375, 0.26407883, 0.27456343))
transforms_student_train_Normalize = transforms.Normalize((0.44738507, 0.41788426, 0.37711918),
                                                          (0.24555556, 0.2418625, 0.25421354))

teacher_norm = transforms.Compose([
    transforms.ToTensor(),
    transforms_teacher_train_Normalize,
])

student_norm = transforms.Compose([
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms_student_train_Normalize,
])

train_dataset = Places_Extra69_Online(split='Training', transform=transform_train, student_norm=student_norm, teacher_norm=teacher_norm)
img_teacher_0, img_student_0, target_0, _ = train_dataset.__getitem__(0)
img_teacher_1000, img_student_1000, target_1000, _ = train_dataset.__getitem__(1000)
img_teacher_51000, img_student_51000, target_51000, _ = train_dataset.__getitem__(51000)
img_teacher_31000, img_student_31000, target_31000, _ = train_dataset.__getitem__(31000)
img_teacher_61000, img_student_61000, target_61000, _ = train_dataset.__getitem__(61000)
img_teacher_98720, img_student_98720, target_98720, _ = train_dataset.__getitem__(98720)

img_teacher = torch.cat((img_teacher_0.unsqueeze(0), img_teacher_1000.unsqueeze(0), img_teacher_51000.unsqueeze(0),
                         img_teacher_31000.unsqueeze(0), img_teacher_61000.unsqueeze(0), img_teacher_98720.unsqueeze(0)), dim=0)
img_student = torch.cat((img_student_0.unsqueeze(0), img_student_1000.unsqueeze(0), img_student_51000.unsqueeze(0),
                         img_student_31000.unsqueeze(0), img_student_61000.unsqueeze(0), img_student_98720.unsqueeze(0)), dim=0)
target = np.array([target_0, target_1000, target_51000, target_31000, target_61000, target_98720])

_, _, _, _, _, logit_t = model_t(img_teacher.cuda())
_, _, _, _, _, logit_s = model_s(img_student.cuda())

tensor1 = temperature[0].unsqueeze(0).unsqueeze(0)
tensor2 = temperature[1000].unsqueeze(0).unsqueeze(0)
tensor3 = temperature[51000].unsqueeze(0).unsqueeze(0)
tensor4 = temperature[31000].unsqueeze(0).unsqueeze(0)
tensor5 = temperature[61000].unsqueeze(0).unsqueeze(0)
tensor6 = temperature[-1].unsqueeze(0).unsqueeze(0)
temperature = torch.cat((tensor1, tensor2, tensor3, tensor4, tensor5, tensor6), dim=0)

# soft_logit_s = torch.div(logit_s, temperature.expand(logit_s.shape))
# soft_logit_t = torch.div(logit_t, temperature.expand(logit_t.shape))
# loss_div = torch.nn.KLDivLoss(reduction="none")(F.log_softmax(soft_logit_s, dim=1), F.softmax(soft_logit_t, dim=1)).sum(1) \
#            * temperature * temperature
# loss_div = loss_div.sum(1).view(1, 6)
#
# plt.figure("loss_div")
# plt.imshow(loss_div.cpu().detach().numpy())
# plt.axis('off')
# if not os.path.isdir('picture/'):
#     os.mkdir('picture/')
# plt.savefig('picture/loss_div.jpg', dpi=500)
# plt.show()



lam = temperature / temperature.max()
lam = lam.expand(logit_s.shape)

mixed_logits_s = (1 - lam) * logit_s + lam * logit_s
mixed_logits_t = (1 - lam) * logit_t + lam * logit_t
loss_mixup = torch.nn.KLDivLoss(reduction="none")(F.log_softmax(mixed_logits_s, dim=1), F.softmax(mixed_logits_t, dim=1)).sum(1)
loss_mixup = loss_mixup.view(1, 6)
print(loss_mixup)
print(loss_mixup.shape)

plt.figure("loss_mixup")
plt.imshow(loss_mixup.cpu().detach().numpy())
plt.axis('off')
if not os.path.isdir('picture/'):
    os.mkdir('picture/')
plt.savefig('picture/loss_mixup.jpg', dpi=500)
plt.show()