from common import *
import subprocess
from os import path


# TODO this script doesn't work


LOREM_IPSUM_TEXT = """
Fusce fermentum justo dictum odio maximus dictum. Morbi lacinia tristique iaculis. In ac dolor tempor, lobortis nulla nec, iaculis risus. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Praesent accumsan metus nec sem ullamcorper aliquam. Morbi egestas enim vel tortor accumsan accumsan ut mattis augue. Pellentesque id varius dolor. Duis bibendum, sem nec vestibulum efficitur, elit turpis commodo leo, a feugiat risus nulla at diam. Interdum et malesuada fames ac ante ipsum primis in faucibus. Duis vel malesuada nunc, in tempus purus. Vivamus ut congue nibh. Suspendisse faucibus urna in rutrum vehicula. Curabitur ut tellus id ligula lacinia feugiat. In mattis diam non ex condimentum rhoncus. Nullam eget ornare ligula, eu convallis sem. Fusce nec sollicitudin elit.

Mauris nulla mauris, vulputate nec sem ut, pulvinar tristique nunc. Vivamus sit amet nibh felis. Nullam libero est, venenatis vitae neque consectetur, vehicula laoreet felis. Etiam luctus aliquam tellus, nec efficitur metus auctor consectetur. Proin pharetra aliquet leo, vel convallis est convallis quis. Sed ipsum eros, porta sed felis ac, porta ullamcorper neque. Nunc vehicula felis sit amet neque vehicula, pulvinar bibendum diam semper. Integer sit amet leo et lectus egestas porttitor. Nam at finibus lectus. Phasellus non rutrum turpis, malesuada bibendum ipsum. Proin ut erat quis felis congue facilisis et vitae neque. Quisque nisi urna, pretium eu felis sed, fringilla tristique leo.

Pellentesque fermentum imperdiet enim a lobortis. Etiam vitae tincidunt nibh. Integer iaculis ipsum id condimentum aliquet. Nunc mi mi, pulvinar vel vehicula id, elementum in arcu. Phasellus fringilla posuere tellus, vitae pharetra velit bibendum vel. Vestibulum malesuada faucibus laoreet. Sed egestas non leo sed commodo. Ut ac molestie arcu. Ut vitae libero eget ante porta pharetra. Donec dictum turpis non ullamcorper consequat. Phasellus congue dapibus eros at aliquet.

Mauris posuere sed lorem a gravida. Etiam mollis odio ante, id fringilla eros scelerisque ac. Sed rhoncus pulvinar nibh ac porta. Praesent aliquam commodo luctus. Maecenas faucibus congue massa eu mattis. Mauris a metus dictum, auctor neque at, scelerisque lorem. Suspendisse ac augue quis sem pharetra viverra non id mi. Curabitur sit amet tempor libero.

Phasellus feugiat magna sed ante ultrices, quis placerat mauris rhoncus. Nam magna magna, interdum tincidunt scelerisque ac, rhoncus vitae est. Donec porttitor nunc et justo rhoncus facilisis. Donec non viverra elit. Suspendisse nec velit eget sapien mattis placerat in ut ante. Aliquam congue mauris sit amet tellus laoreet volutpat eu et neque. Ut lacus diam, efficitur at sem id, pellentesque tempor leo. Mauris augue elit, maximus vitae mi nec, mattis egestas mauris.

Donec ultrices tortor diam, ac mattis dolor cursus ac. Pellentesque nulla neque, consectetur eu nunc in, blandit lacinia lacus. Mauris eros felis, lobortis a pellentesque quis, iaculis non elit. Nulla viverra enim malesuada nulla tristique tincidunt vel id magna. In erat velit, volutpat et felis nec, commodo sodales massa. Nam dignissim lorem at orci pretium interdum sit amet a massa. Etiam sit amet viverra nisi. Donec vitae nisl nec massa porta pellentesque. Integer molestie vestibulum dui sit amet mattis. Morbi ut maximus tortor.

Proin elit felis, dictum quis justo et, venenatis fringilla lectus. Aliquam efficitur efficitur mi. Praesent pharetra rhoncus ornare. Nulla sed iaculis risus. Quisque massa felis, dignissim eget nibh et, fringilla pellentesque magna. Donec congue lorem non sem volutpat, sit amet accumsan ante iaculis. Nunc ut efficitur tortor. Nunc a posuere nisl, et pulvinar metus. Proin varius, sem eu sodales mollis, sapien neque varius odio, et aliquam ex enim viverra justo. Morbi consequat ultrices libero, a mollis tortor congue ut. Quisque feugiat nunc non orci lobortis, sed tempus turpis vehicula. Praesent nec tellus et risus faucibus porttitor.

Sed scelerisque, nisi ac ultricies lacinia, nisi est mattis magna, ut pellentesque lorem diam in erat. In mauris sem, iaculis quis efficitur vitae, cursus quis sapien. Fusce nec diam convallis, fermentum neque ac, convallis mauris. Nunc viverra felis lacus, sit amet pellentesque dolor dignissim quis. Vestibulum hendrerit suscipit varius. Morbi ac pharetra libero. Sed luctus massa vel euismod pretium. Quisque nibh risus, pulvinar sit amet consequat non, lacinia id orci.

Donec iaculis non enim sit amet aliquam. Vestibulum ante ipsum primis in faucibus orci luctus et ultrices posuere cubilia curae; Aenean nec metus ornare, rutrum lacus sed, tincidunt lorem. Phasellus lorem nibh, mollis quis varius ac, malesuada et diam. Nam faucibus turpis in metus sollicitudin, eu pulvinar orci sollicitudin. Suspendisse vel purus non libero molestie sodales. Nullam in pulvinar justo, ac sodales nibh. Suspendisse laoreet nisl vel diam fringilla, eget ullamcorper elit aliquet. Maecenas nec pulvinar ipsum, vel ullamcorper diam. Praesent interdum mauris quis tempor porta. Aliquam eget nisi orci. Nullam bibendum euismod metus, ut tristique elit feugiat eget. Praesent fermentum diam id mi blandit, et consequat quam fermentum. Cras quis ante ipsum. Orci varius natoque penatibus et magnis dis parturient montes, nascetur ridiculus mus. Fusce vel orci eget purus ultrices cursus.

Quisque mattis purus arcu, sit amet varius enim ultricies in. Vestibulum sit amet diam at ante placerat aliquet. Proin vitae pretium nulla, et egestas ipsum. Aenean ultrices massa a dolor laoreet, et auctor velit vestibulum. Etiam tempor consectetur mauris, id ullamcorper nisl. Curabitur hendrerit libero at mauris placerat laoreet. Cras interdum, nunc condimentum semper efficitur, ipsum neque ornare urna, id iaculis arcu enim sit amet purus. Sed iaculis pretium libero quis auctor. Vivamus auctor porttitor velit ac ornare. Ut porttitor velit ut metus mattis ornare. Suspendisse laoreet justo vel dui porttitor dapibus. Vestibulum consectetur eget erat mattis tincidunt.

Sed eu semper sem, sit amet aliquam turpis. Duis dictum felis ac dui dapibus tempus. Cras tincidunt lacus non tincidunt accumsan. Aliquam laoreet metus in vestibulum luctus. Duis vitae risus molestie, pretium tortor euismod, facilisis augue. Nam vitae orci eu tortor porta dictum. Sed ut tincidunt leo. Vestibulum bibendum neque eget elit aliquet rhoncus. Integer mauris metus, suscipit vel odio quis, congue consequat metus.

Fusce consectetur elit eu luctus scelerisque. Donec id justo sem. Nam sit amet interdum purus. Nam posuere velit vel nunc blandit tempus. Curabitur laoreet libero mi, ut cursus est feugiat vel. Donec lobortis porttitor erat, a porttitor sapien.
"""


def _main():
    if not path.exists(FSC_CLASSIFY_DATASET_DIR):
        os.mkdir(FSC_CLASSIFY_DATASET_DIR)

    # Regular
    with open(path.join(FSC_CLASSIFY_DATASET_DIR, f"regular.md"), "w") as f:
        txt = " ".join(LOREM_IPSUM_TEXT.split())
        f.write(txt)

    # Bold
    with open(path.join(FSC_CLASSIFY_DATASET_DIR, f"bold.md"), "w") as f:
        txt = " ".join([f"**{x}**" for x in LOREM_IPSUM_TEXT.split()])
        f.write(txt)

    # Italic
    with open(path.join(FSC_CLASSIFY_DATASET_DIR, f"italic.md"), "w") as f:
        txt = " ".join([f"*{x}*" for x in LOREM_IPSUM_TEXT.split()])
        f.write(txt)

    # Run md-renderer from its working directory
    cwd = path.realpath(path.join(path.dirname(path.realpath(__file__)), "../md-renderer"))
    subprocess.Popen(["sudo", "bash", "./main.sh", FSC_CLASSIFY_DATASET_DIR], cwd=cwd)


# TODO delete *-bb-* files inside the dataset folder


if __name__ == "__main__":
    _main()

