# Contribution Rules

## Coding Guidelines

- Please follow the existing conventions in the relevant file, module and project when you add new code or when you extend/fix existing functionality.

- Try to keep pull requests (PRs) as concise as possible:
    - Avoid committing commented-out code.
    - Wherever possible, each PR should address a single concern.

- Write commit titles using imperative mood and [the following rules](https://chris.beams.io/posts/git-commit/), Use the following template:
```
[<Changes Component>] <Title>

#<Issue Number>

<Commit Body>
```
- Ensure that the build log is clean, meaning no warnings or errors should be present.

## Pull Requests
Developer workflow for code contributing:

1. Developers must first [fork](https://help.github.com/en/articles/fork-a-repo) the upstream [rivermax-dev-kit](https://github.com/NVIDIA/rivermax-dev-kit) repository.

2. Git clone the forked repository and push changes to the personal fork.
```bash
$ git clone http://github.com/YOUR_USERNAME/YOUR_FORK.git
# Checkout the targeted branch and commit changes
# Push the commits to a branch on the fork (remote)
$ git push -u origin <local-branch>:<remote-branch>
```

3. Once the code changes are staged on the fork and ready to review, a Pull Request (PR) can be requested to merge the changes from a branch of the fork into a selected branch of upstream.
    * Exercise caution when selecting the source and target branches for the PR.

## Signing Your Work

- We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.
    - Any contribution which contains commits that are not Signed-Off will not be accepted.
- To sign off on a commit you simply use the `--signoff` (or `-s`) option when committing your changes:
``` bash
$ git commit -s -m "Add cool feature"
```
This will append the following to your commit message:
```
Signed-off-by: Your Name <your@email.com>
```
- By doing this you certify the following:
```
Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.
1 Letterman Drive
Suite D4700
San Francisco, CA, 94129

Everyone is permitted to copy and distribute verbatim copies of this license
document, but changing it is not allowed.

Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:
(a) The contribution was created in whole or in part by me and I have the right to submit it under the open source license
    indicated in the file; or
(b) The contribution is based upon previous work that, to the best of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that work with modifications, whether created in whole or in part by
    me, under the same open source license (unless I am permitted to submit under a different license), as indicated in the file;
    or
(c) The contribution was provided directly to me by some other person who certified (a), (b) or (c). and I have not modified it.
(d) I understand and agree that this project and the contribution are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is maintained indefinitely and may be redistributed consistent
    with this project or the open source license(s) involved.
```
