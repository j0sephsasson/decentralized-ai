async function buy(clicked) {
    App.web3 = new Web3(window.ethereum);
    accounts = await ethereum.request({
        method: 'eth_requestAccounts'
    });
    App.start();

    document.getElementById('buyBTN-d').style.display = 'none';
    document.getElementById('buy-progress').style.display = 'initial';

    var price = await App.meta.methods.checkPrice(clicked).call();

    var etherValue = App.web3.utils.fromWei(price.toString(), 'ether');
    var dojoETHFee = Number(etherValue) * 0.05;
    var dojoWEIfee = App.web3.utils.toWei(dojoETHFee.toString(), 'ether');

    resp = await App.meta.methods.buyModel(clicked).send({
        from: accounts[0],
        value: Number(price) + Number(dojoWEIfee)
    });

    $.post("/buy_item", {
        buyer: accounts[0],
        item: clicked
    }, function (err, req, resp) {
        document.getElementById('buy-progress').style.display = 'none';
        document.getElementById('buyBTN-d').style.display = 'initial';
        window.location.href = "/profile"
    });
};

// white paper alert for MVP
function whitePaper() {
    alert("White Paper Coming Soon!");
};