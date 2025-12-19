import batbot


def test_fetch():
    batbot.fetch(pull=False)
    batbot.fetch(pull=True)

    batbot.fetch(pull=False, config='usgs')
    batbot.fetch(pull=True, config='usgs')


def test_example():
    batbot.example()
