# How to Contribute

## Coding Convention and Requirements

* Write object-oriented code.
  - This has the advantages of data hiding and modularity. It allows reusability, modularity, polymorphism, data encapsulation, and inheritance. Also, you should write modular and non-repetitive code. Use classes and functions in your code.
* Write the code according to PEP 8. For more information, read the [PEP 8](https://www.python.org/dev/peps/pep-0008/) page.
  - E.g: PyeIQ uses *snake_case* for variable names, function names, method names, and module or package names.
* Each feature should come with test cases that can be executed as unit tests during build.
* Each feature should come with small codes for executing the classes.
* Avoid as much as you can adding dependencies that are not currently supported by NXP BSP.
  - This must be handle as optional feature using _accert_ or _try_ methods to verify if it is installed or not.
* The submitter has the first responsibility of keeping the created pull request or patch:
  - Clean and neat, which means that must be readable;
  - Must not apply general rules of git commits and common senses;
  - Must not write a lengthy commit; must have a single topic per commit;
  - Must provide enough background information and references.

## Signing-off Commits

* Each commit is required to be *signed-off* by the corresponding author:
  - Properly configure your development environment, you can add sign-off for a
commit with *-s* option: e.g., *git commit -s*;
  - Make sure to configure your *.gitconfig* file correctly:
```bash
$ vi ~/.gitconfig
  [user]
          name = Your Name
          email = Your_Email@nxp.com
$ git commit -s <file_name>
// -s (--signoff) means automated signed-off-by statement.
```
Including a "Signed-off-by:" tag in your commit means that you are making the Developer Certificate of Origin (DCO) certification for that commit. A copy of the DCO text can be found at [Developer Certificate](https://developercertificate.org/).

## Code Reviews and Pull Requests/Patches

* The patches or pull requests are reviewed by the maintainers, committers and reviewers.
* If you are a committer or a reviewer of the specific component, you are:
  - **Obligated** to review incoming related pull requests or patches.
  - **Give** feedback on pull requests or patches especially on similar topics/components.

### Merge Criteria

A pull request must be according to the following statements to be accepted:
* Passed all the tests executed for the any other code reviewer.
  - This includes unit tests and integration tests;
  - If the pull requests affects sensitive codes or may affect wide ranges of
    components, reviewers will wait for other reviewers to back them up;
  - If the pull request is messy, you will need to wait indefinitely to get reviews.
* There is no rejections from any official reviewers.
* There is no pending negative feed-backs (unresolved issues) from reviewers.
* A committer with merging privilege will, then, be able to merge the given pull request.
