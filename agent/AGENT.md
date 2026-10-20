# Use case

- **Pitch:** a scholarship reviewer that reads a student's application reason and decides whether
  it matches the scholarship policy, returning eligible / needs_review / ineligible with a short
  explanation.
- **Who it is for:** a university scholarship office, so staff are not reading every application by
  hand; unsure cases still go to a human.
- **Features:**
  - reads a student's free-text application reason against the Need-Based Financial Aid Scholarship
    policy;
  - classifies `eligible` / `needs_review` / `ineligible` with calibrated confidence;
  - low-confidence or vague reasons are held for a human, never auto-approved;
  - a short, plain explanation that references the policy clause;
  - look up an application by student id from sample data.
- **The decision handed to Strands Decider 2B:** does the applicant's reason match the scholarship
  policy? (`#policy`)
- **Why a decision model:** the check is a language judgement (does this written reason satisfy a
  policy written in prose), it must return a confidence the code can branch on, and it runs on
  every application.
- **Local only:** this build runs locally (`.\kit.cmd run` / `.\kit.cmd demo`); it is not deployed.
- **Built with the AWS User Group Madurai Strands Decider hackathon kit.**
